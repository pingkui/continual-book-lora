# Continual Book Learning with LoRA

Sequential knowledge-injection experiments with **Qwen2.5-7B-Instruct**: learn four synthetic Russian fact books while measuring forgetting, false-premise handling, and general-capability drift. The model answers from merged weights, without retrieval or per-book adapter switching.

Research code, recorded results, and failed experiments are included. General-capability preservation remains unresolved.

## Method

- **Projected LoRA:** rank-32 adapters on seven attention/MLP projection types, merged after each book. Protected activation subspaces cover earlier books, general Russian text, and self-generated instructions.
- **Spaced review:** self-tests identify forgotten facts; their training rows are reviewed sooner, while correctly recalled facts are revisited at increasing intervals.
- **Generated training data:** the base model produces paraphrases, questions, and true/false-premise pairs. The `answer_first` format recalls facts before correcting a false premise.

Implemented in JAX, with cached generation, checkpointing, Hugging Face export, and numerical checks against PyTorch/Transformers.

## Measured results

Results use the synthetic testbed's main exam: 150 knowledge questions per book (75 direct, 75 indirect), with held-out wording. Client books and blind acceptance tests have not been evaluated.

### Four-book sequence, two seeds

Book 1 uses reduced row types for 6 epochs; books 2–4 use full row types for 12 epochs.

| Run | Books 1/2/3 just after learning | Books 1/2/3/4 after book 4 | Retention of books 1/2/3 |
|---|---|---|---|
| [gen24, seed 0](tpu/results/gen24-20261029/) | 142 / 147 / 144 | 143 / 147 / 141 / 143 | 100.0% / 99.3% / 97.9% |
| [gen25, seed 1](tpu/results/gen25-20261029/) | 146 / 149 / 145 | 143 / 144 / 142 / 143 | 97.3% / 96.6% / 97.2% |

Retention is the fraction of **previously correct answers that remain correct**.

Strict Transformers re-scoring after book 4 catches **59/58/60/60** traps for seed 0 and **60/57/60/59** for seed 1, out of 60 per book. Both runs leave all 30 true-premise twins per book unrefused. A caught trap must deny and name the false value; seed 1 misses the 58/60 target on book 2.

Only **5 of 8** book-by-seed checkpoints meet local L5 limits: each benchmark drops ≤1 percentage point, summed drops ≤1.5. Evaluation uses IFEval prompt-level strict, GSM8K 5-shot strict-match, and full MMLU 5-shot through `lm-evaluation-harness`.

A separate, newer book-1 experiment ([gen31](tpu/results/gen31-20261030/), [gen32](tpu/results/gen32-20261030/), full row types, 5 epochs) scores **148/150 in both seeds** and meets book-1 L5 limits. It has not yet been continued through books 2–4.

## Scope and reproduction

An earlier book-4 run reduced peak JAX device memory from **52.9 to 32.4 GiB** using half-precision bases and gradient checkpointing. Measured on a 96 GB GPU with a constrained JAX budget; 48 GB A40/A6000 operation remains unverified, and that run did not pass L5.

Weights, LoRA factors, training-row archives, and offline dependency bundles are excluded. Scripts retain Colab/Linux paths, and the package launcher needs alignment with the latest configuration; this snapshot is not a turnkey reproduction package.

## Where to look

- [Training implementation](tpu/jax/) and [numerical checks](tpu/jax/tests/).
- [Synthetic books and exams](lab/data_v18/), [strict scorer](lab/score_strict.py), and [recorded results](tpu/results/).
- [Detailed experiment history](tpu/JAX_PLAN.md) (Chinese), including ablations and evaluation-setting comparisons; [earlier 1.5B report](lab/REPORT.md) (English).
- [Experimental package interface](tpu/jax/package/README.md): one `run.sh BOOK.json OUT_DIR` call per book, exporting a merged checkpoint. See the reproduction limits above before running.
