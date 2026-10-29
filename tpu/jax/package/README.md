# Continual book learning for Qwen2.5-7B-Instruct, one 48 GB card

    bash run.sh BOOK.json OUT_DIR            # one call per book, in order, the same OUT_DIR for the whole run
    bash run.sh BOOK.json OUT_DIR --resume   # after an infrastructure failure: continue the book in progress

Input: a JSON (or JSON-lines) file of Russian fact units. The loader (code/generic.py, `load_units`) takes the longest
list of strings or objects anywhere in the document (or a dict keyed by id); an object's text is its first non-empty
field among text / unit / fact / sentence / content / statement / passage / ru / text_ru / value / body, else its
longest string value; its id is the first of id / uid / unit_id / fact_id / key / idx / index / n / number, else the
position. Output after every call: `OUT_DIR/model`, one set of merged weights in the base architecture
(Qwen2ForCausalLM, same tensor shapes as Qwen2.5-7B-Instruct, safetensors + config + tokenizer, no remote code), for
`transformers` greedy decoding with the model's own chat template. Nothing is switched or retrieved at answer time.

## What one call does (details in code/run_cl.py)

1. Environment (first call only): the package's own CPython 3.12 (env/python.tar.gz) is unpacked into OUT_DIR/pyroot
   and the pinned wheels (env/wheels, env/requirements.txt) are installed into it with `pip --no-index`. No network. Then the pinned base is downloaded from Hugging
   Face into OUT_DIR/base (the only network use of the whole run).
2. The base model itself writes the training material for the book (code/generic.py): paraphrases, questions,
   single-detail falsifications and premise questions (true and false) for every unit. No external teacher, no API.
3. One LoRA (rank 32, seven projections) is trained on the book and merged into the weights. Every update is projected
   off the layer-input subspaces of the earlier books, of general Russian text (data/general_ru.json, shipped) and of
   320 instructions the base model wrote and answered itself at the first call.
4. From the second book on, earlier books are reviewed with a per-epoch self-test (spaced review, rules below).
5. The merged weights are written to OUT_DIR/model and rebuilt from the saved per-book factors (OUT_DIR/lora), so the
   weights are reproducible from the factors.

## Rules (every decision can be recomputed from run.log and teacher.log)

- Epochs of book k: `E = 6 if N_k >= 1000 else 12`, N_k = units of book k (run.log "units").
- Learning rate: 2e-4, cosine over `E * ceil(rows / 32)` steps, batch 32 as 4 micro-batches of 8; seed `k`.
- Rows of book k (generic.py): `text` 1 per unit; `text_g` up to 3 paraphrases that keep the detail; `qa` 2 (direct
  question, inverse question); `qa_g` up to 2 teacher questions; `verify` 1 true + up to 3 false; `premise_g`: partners
  of a unit = units sharing a proper name (names stemmed by one trailing vowel; a name shared by more than 8 units is
  not an entity) or the teacher's subject key, at most `P = 10` partners, one true premise question per pair and one
  false premise question per falsification of the partner. Falsifications: up to 2 swaps (the same sentence skeleton
  with another unit's value) then teacher ones that keep the unit's main entity, 3 at most.
- Protection: the book subspace of every projected module keeps the top directions of the layer-input covariance up
  to `tau = 0.99` of the energy; general text `tau_gen = 0.70` over 300 shipped texts; self-instruct `tau_chat = 0.90`
  over 320 rows. Updates are projected off the union (fp16 bases).
- Spaced review (book k >= 2): after every epoch the model is tested on `min(300, |facts|)` sampled facts of each
  earlier book; a fact answered wrong is reviewed in the next epoch, a fact answered right is reviewed again after
  `2^m` epochs (m = consecutive right answers); the review rows of an epoch are capped at `1.0 x` the new book's rows.
- Budget guard (section 7): training steps stop at `T_start + (BUDGET_H - RESERVE_H)` with `BUDGET_H = 8`,
  `RESERVE_H = 1` (run.log "budget guard"); the book is then finished with the factors trained so far (self-test,
  subspace, saving, export all run), so the call ends inside 8 card hours. Teacher budget: every teacher call is
  charged before it runs at `TEACHER_PRICE_PER_1K = 0` USD per 1k generated tokens against `TEACHER_BUDGET_USD = 20`;
  a call that would exceed the cap raises, no further teacher call is made for the book, and training proceeds on the
  rows written so far. With the local teacher the spend is 0.
- Checkpoints: an adapter checkpoint at step 200 of the book (OUT_DIR/ckpt/.../book{k}.early.npz) and after every
  epoch (book{k}.npz, the --resume point); removed when the book is saved.

## Teacher

The teacher is the pinned base itself, running in the same process (JAX, greedy, batch 16). OUT_DIR/teacher.log has
one JSON line per call: prompts, answered, max tokens, generated tokens, seconds, cost (0), running spend, and one
sample prompt/answer. No Anthropic Claude model is involved anywhere; no row is written by anything but the pinned base.

## Resources

48 GB card: fp16 protection bases, gradient checkpointing, micro-batch 8, generation batch 48; JAX preallocates 75% of
the card (36 GB), torch's eigendecompositions take the rest (on the GPU when its build has kernels for the card, else on
the CPU). Measured on a 96 GB card capped to a 40 GiB JAX budget: peak 30-33 GiB. Time on that card (0.86 s/step): a
2,176-unit book about 2.5 h including 35 min of generation; small books 15-40 min; the first call also installs the
packages (about 3 min), downloads the base (about 15 GB) and writes the 320 self-instruct rows (about 7 min).

## Files

run.sh, env/ (python.tar.gz, wheels/, requirements.txt), code/ (JAX training and generation, export to a HF
checkpoint), data/general_ru.json, LICENSES.md, MANIFEST.txt (sha256 of every file and the commit). Training rows are
written under OUT_DIR/src and never contain IFEval, GSM8K or MMLU text (they are written in Russian from the book's own
sentences; the self-instruct rows are the base model's answers to generic Russian and English seed instructions).
