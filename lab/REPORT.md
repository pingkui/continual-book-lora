# Fact-injection experiments: measured results (synthetic testbed)

All numbers come from Kaggle T4 runs of `lab/` scripts. Setup: Qwen2.5-1.5B-Instruct (NOT the 7B target),
templated fictional Russian "books" (cities/founders/rivers/goods/population/year), books of 119-145 units
(book 1 scale, 2,176 units, not tried), LoRA r=16, 15 epochs, one seed unless stated. Exam wordings are held out
from training. Knowledge = direct+indirect correct out of 150 per book.

## Q1: false-premise traps (target: <=2 made-up of 60)
| training | trap made-up /60 | true-twin false refusals /30 |
|---|---|---|
| base model | 8-11 | 24-26 |
| book text + QA only | 60 | ~0-3 |
| + "is it true that" contrast pairs | 60 (no effect) | ~0-3 |
| + false-premise correction rows | 34-44 | 3-10 |
| + recall-then-compare rows (rounds 3-4) | 0-25 as scored, NOT VALID | see below |

**Scoring flaw found in round 4 (all recall-row numbers above are unreliable).** The twin scorer counted an answer as
correct if it contained the gold value, even when it was followed by "Это неверно: ...". The recall-trained models
learned to append that phrase almost always: among sampled twin answers, 29/36 (A2_recall2) and 12/36 (A_recall) were
"correct" by the old scorer but contained a false correction, with an invented number ("указанное в вопросе — 1693").
So their low trap counts (A2: 0-2/60) reflect near-unconditional denial, not premise checking. The scorer is fixed
(a denial on a twin is now a false refusal). Round-5 rerun of A2 saved full answers (the Kaggle run itself still used the old
scorer because the dataset carried the old factlab.py; rescored locally with the fixed one):
| A2_recall2, strict scoring | traps made up /60 | twin correct / false refusal /30 |
|---|---|---|
| book 2 after book 2 | 0 | 21 / 9 |
| book 3 after book 3 | 4 | 22 / 8 |
| book 4 after book 4 | 2 | 19 / 11 |
So on a freshly learned book the recall rows reach L1 on traps (<=2/60 made up, 0-4 across stages) but over-refuse true questions
in 8-11/30 cases (line: <=2/30), and old books get worse (13-18/30). That is a trade-off between the two lines, not a pass.
Older runs (A_recall, D_combo, earlier) could not be rescored.

**Audit of A2 answers (round 5).** Per trap type, on every stage: founder_swap ("В каком году <имя> основал город X?") is denied
for traps 18-20/20 AND twins 7-10/10, i.e. pure template denial with no checking, and in those denials the value "stated in the
question" is invented (a year, when the question named a person) in 18-20/60 trap answers. river_swap and year_swap discriminate
on a freshly learned book: traps 18-20/20 denied, twins 0-1/10 denied (older books: up to 5/10). So the recall rows generalise to
premise clauses in the trained position, not to a premise moved into the subject of the question.
Matcher fix: gold names used to match on a cut surname stem, so 13-26 founders per book were accepted for another founder
("Вардан" vs "Вардский"). Now every word must appear (one final vowel/й/ь may change). Rescoring A2 changed little (-2 knowledge
items, -1 twin). Remaining collisions only in book 1 (10/435 rivers, 4/436 cities) and мёд/медь.
The earlier rows (qa only, verify, plain premise rows, projection runs) had no such pattern in their samples.

**Round 6: premise in three frames (trailing clause / own sentence first / "Учитывая, что ..."), true:false = 1:1 (A3).**
Strict scoring. Fresh book: traps 36-40/60 made up, twins 29-30/30 correct (0-1 false refusals). Per type, traps denied:
year_swap 20/20 with twins 0/10 denied (real discrimination); founder_swap 0-2/20; river_swap 0-3/20. In river_swap traps the
model often states the TRUE river after answering ("... — свинец. Через город Варденка протекает река Зелурена.") but never flags
the contradiction. year_swap's exam clause ("основанного в Y году") is close to a trained clause ("основанный в Y году", other
asked relations), so its success is likely near-template transfer. A2 caught river_swap but not founder_swap; A3 catches neither.
Conclusion: on templated data at 1.5B the recall rows teach phrasing-specific detectors, not premise checking; balancing
true/false removes over-refusal and the trap gains with it.

**Round 7: model-written templates (A4) vs code templates (A3), two exams.** Teacher Qwen2.5-7B wrote statement /
question / premise templates (269 premise templates over 17 pairs), Vikhr-Nemo-12B wrote a second exam (120 traps over 3
never-trained pairs + 3 trained pairs, 60 twins that reuse the trap's template with the true value) and judged all templates.
Freshly learned book, strict scoring:
| | old exam traps /60 | old twins ok / refused /30 | new exam traps /120 | new twins ok / refused /60 | new-exam knowledge /150 |
|---|---|---|---|---|---|
| A3 (code templates) | 3-5 | 21-24 / 6-9 | 6-9 | 32-41 / 15-27 | 120-136 |
| A4 (model templates) | 2-11 | 29-30 / 0-1 | 3-7 | 55-58 / 2-4 | 135-141 |
On the new exam A4 denies 13-20/20 traps in every never-trained pair while refusing 0-2/10 twins per pair, and its denials
cite the value actually in the question (114/117), i.e. it recalls, compares and then corrects; A3's low trap count comes with
indiscriminate denial (13 invented cited values, 15-27/60 refused twins). Weak spots: A4 on the old exam's river_swap
(9-13/20 denied on books 3-4) and the never-trained pair founded|founder (13-17/20).
**Seed sensitivity:** A3 in round 6 and round 7 differ only in row order (and 1-3 dropped QA rows), yet old-exam traps went
from 36-40/60 to 3-6/60 and twin refusals from 0-1 to 6-9/30. Single-seed trap numbers are therefore not reliable.
**Round 8: A4 with seeds 1 and 2.** Freshly learned book (books 2/3/4 right after their own training), new exam:
| | traps made up /120 | twins refused /60 | knowledge /150 | invented cited values /120 |
|---|---|---|---|---|
| A4 seed 0 | 7, 5, 3 | 4, 2, 3 | 141, 138, 135 | 3, 3, 3 |
| A4 seed 1 | 7, 5, 1 | 6, 10, 7 | 144, 147, 137 | 10, 10, 5 |
| A4 seed 2 | 6, 8, 2 | 4, 9, 7 | 148, 145, 139 | 10, 12, 10 |
| A3 (round 7) | 6, 9, 7 | 15, 26, 27 | 136, 125, 120 | 8, 18, 13 |
All three A4 seeds keep traps low and refuse far fewer true twins than A3, and recall more on the other model's wording:
the gain from model-written data holds across seeds. It is near the client's lines (scaled: <=4/120 traps, <=4/60
refusals) but not reliably under them (seeds 1-2 refuse 6-10/60). On the OLD exam (code-template wording, which A4 never
trained on) seed 1 makes up 18-22/60 traps vs 2-11 for seeds 0 and 2, so robustness to an unseen style is still seed-
dependent. Older books: twin refusals climb to 21-27/60 as their knowledge decays (71-78/150 after book 4): forgetting now
drives the remaining refusal errors.
Unchanged: retention (book 2 after book 4: A4 58% / 52%, A3 55% / 51%), and general-text NLL rises without protection
(A4 3.6-4.1, A3 3.4-3.7, base 1.88).

## Q2: sequential books 2 -> 3 -> 4, retention of book 2 knowledge (150 = all kept)
| method | after book 3 | after book 4 |
|---|---|---|
| plain LoRA, no merge | 21 | 9 |
| merge per book, no projection | 28 (19%) | 16 (11%) |
| null-space projection tau=0.90 | 93 (62%) | 69 (46%) |
| projection tau=0.99 (seed 0) | 123 (82%) | 78 (52%) |
| projection tau=0.99 (seed 1) | 113 (75%) | 76 (51%) |
| projection tau=0.999 | 120 (80%) | 75 (50%) |
Target is >=95%. Seed 0 vs 1 differ by ~10 points after book 3, so differences between tau values
(0.99 vs 0.999) are within noise. New books are still learned (146-150/150).

## Round 3 (recall rows; MLP-only; general-text protection), one seed each
- Recall rows (A_recall): book 2 know 135/150, retention of book 2 after book 3 / 4: 69% / 49% (same as projection alone).
- MLP-only, 8 epochs, general-text subspace protected: the model barely learned (36-39/150 on book 2, traps 60/60).
  Three changes were stacked, so the cause is unattributed. Retention figures from these runs are meaningless.
- General-text NLL (Wikipedia-ru; base 1.88): protected runs 2.0-2.4, unprotected A_recall 3.5-3.7.
- Round 4: MLP-only 15 epochs (B1): book 2 know 136/150, retention after book 3 / 4: 70% / 54%, traps 41-47/60 made up
  (same as the all-modules projection runs, so MLP-only alone changes little). Recall rows with two false premises (A2):
  retention 80% / 62% and traps 0-2/60, but see the scoring flaw above. General-text protection run (B2) crashed in the
  eigh step (GPU OOM); fixed by falling back to CPU, not rerun yet.

## Round 6: general-text protection from the start (B2, all modules, 15 epochs, fp16 bases)
General-text NLL stays 2.07-2.23 (base 1.88; unprotected runs 3.5-3.8), new books still learned (127-138/150), but book 2 retention
after book 3 / 4 falls to 42% / 11% (projection without it: ~80% / ~55%). Protected dims per module reach 1264-1785 on average,
above the 1536 input width of most layers, so later books can only use low-energy leftover directions. Hypothesis, not checked:
large weights on those directions amplify old inputs' small residuals there. Leakage with fp16 bases 2.5e-4 to 4.2e-4.

## Diagnostic: is the remaining forgetting only a wording shift?
Asked the same facts with the TRAINING wording (tau=0.99, seed 1): book 2 after book 3: 108/150 (exam wording 113);
after book 4: 87/150 (exam wording 76). Same loss either way, so the facts really are overwritten; the
"subspace estimated on training inputs misses new wordings" explanation is not supported.

## Negative / unfinished
- Contrast ("verify") pairs: no effect on traps.
- Projection does not touch trap behavior (independent problem).
- Higher tau: more protected dims (mean ~100 -> ~1500) and more twin refusals (up to 10/30), no retention gain.
- Seed-0 repeat of the diagnostic run crashed (`linalg.eigh` failed to converge); not rerun.
- Not measured: 7B, 2,176-unit book, IFEval/GSM8K/MMLU drift, client's real exam, more than 2 seeds.

---

# Part 2 (2026-10-24 to 10-26): full order 1 -> 2 -> 3 -> 4 on Colab TPU/A100, `tpu/jax/`

Everything in Part 1 above ran on books 2-4 only, with plain replay-free projection. Part 2 is a JAX rewrite of the
pipeline (`tpu/jax/run_cl.py`) run on a Colab TPU v5e-1 and A100 40 GB, still on the **Qwen2.5-1.5B-Instruct proxy**
(not the 7B), still on templated fictional Russian books with the same exams. Book 1 is the client-scale book
(2,176 units, 44,378 training rows for the configuration used); books 2-4 have 145 / 119 / ~130 units. Per book: 150
knowledge questions (75 direct + 75 indirect) in held-out wordings, 60 false-premise traps, 30 true twins; a second exam
written by a different model (Vikhr-Nemo-12B) adds 150 knowledge questions, 120 traps and 60 twins ("exam_gen").
Scoring is strict (a denial on a true twin is a false refusal). One seed (0) unless stated.

## Method (the combination, none of its parts is on the client's dead-ends list as used here)
- LoRA on all seven projections (rank 16, later 32; alpha = 2 x rank), one adapter per book, **merged into the weights
  after each book**, so the result is one Qwen2ForCausalLM weight set per book; nothing is switched at answer time.
- **Null-space projection** of the LoRA update against three protected activation subspaces: (a) each learned book's
  subspace (tau 0.99 of the energy of 4,096 sampled token activations per layer and projection), (b) a general-text
  subspace (300 Russian Wikipedia rows, tau 0.70), (c) a **self-instruct subspace**: 320 instructions the base model
  wrote itself from seed prompts (formatting tasks, grade-school math word problems, multiple-choice questions,
  explanations; EN and RU) and answered itself; tau 0.80, later 0.90; applied to the first book only. No benchmark
  item is used anywhere.
- **Spaced review with self-test** instead of replay: after every epoch the model is asked the earlier books' facts
  (two training wordings each); a wrong fact goes to box 0 and is reviewed next epoch with 6 of its own training rows, a
  right one moves up a box and is next reviewed 2^box epochs later. Review rows per epoch are capped at 1x the new
  book's rows; the self-test covers the reviewed facts plus 300 sampled others. With 2,176 earlier facts this is ~520
  reviewed facts (~2,950 rows) per epoch.
- **Data**: statement / question / premise templates written by a teacher (Qwen2.5-7B-Instruct), judged by
  Vikhr-Nemo-12B, filtered by rules (`tpl_check.py`), with every exam wording and near-copy (Jaccard >= 0.75) removed;
  all 20 (asked relation, premise relation) pairs are trained, including the three the exam traps use.
- Numerics: bases re-orthonormalised by QR with a full |Q^T Q - I| check (a non-orthonormal basis from a failed SVD had
  doubled the LoRA every step in one projection kind).

## Results after book 4, main exam (client lines: knowledge >= 135/150, retention >= 0.95, traps <= 2/60, twins <= 2/30)
| configuration | knowledge b1/b2/b3/b4 | retention b1/b2/b3 | traps made up /60 | twin false refusals /30 |
|---|---|---|---|---|
| rank 16, 12 epochs (round 19) | 131 / 136 / 139 / 134 | 1.000 / 1.000 / 1.000 | 19 / 18 / 19 / 20 | 0 / 1 / 0 / 0 |
| rank 32, 12 epochs (round 20) | 150 / 150 / 149 / 149 | 1.000 / 1.000 / 1.000 | 5 / 8 / 2 / 6 | 0 / 0 / 0 / 0 |
| rank 32, book 1 at 6 epochs (round 21) | 131 / 135 / 138 / 136 | 1.000 / 0.993 / 1.000 | 0 / 1 / 0 / 0 | 0 / 0 / 0 / 0 |
| + 40 more founder-question templates, self-instruct tau 0.90 (round 22) | **150 / 150 / 150 / 150** | **1.000 / 1.000 / 1.000** | **0 / 1 / 1 / 0** | **0 / 0 / 1 / 1** |
Second exam (exam_gen), round 22, knowledge / traps made up of 120 / twin refusals of 60: 148/3/0, 147/3/3, 147/2/3,
145/5/3.

What each step did:
- Retention was 1.000 for every earlier book in every configuration: projection + spaced review hold at 2,176 facts.
- Rank 16 -> 32 fixed knowledge (131 -> 150) and most of the traps (19-20 -> 2-8): at rank 16 the big first book
  learned "a founder's name in a year question means answer the year", and the projection then froze that behaviour
  for all later books ("what the first book learns is final").
- 12 -> 6 epochs on book 1 fixed the remaining traps (0-1 per book, no drift after later books) and lowered the L5
  damage, at the cost of one knowledge wording ("Кому принадлежит основание города X?", 19/75 direct questions
  answered with the year); more epochs had learned that wording but at the L5/trap cost.
- Adding 40 teacher-written founder-question templates with that syntactic variety (noun "основание", dative
  constructions, imperatives) made the wording learnable in 6 epochs (56 -> 75/75) for all four books.

## L5 (lm-evaluation-harness 0.4.13 @ ddd67220, bf16 greedy; 1.5B proxy; drops vs the base model)
| configuration | IFEval prompt-strict | GSM8K flexible | MMLU | sum of drops (line 1.5) |
|---|---|---|---|---|
| base model | 39.74 | 50.42 | 60.23 | |
| small books only, no big book (round 17) | -3.0 | -0.7 | -0.6 | 4.3 |
| rank 16, 12 ep, after book 3 | -0.7 | -6.5 | -1.8 | 9.0 |
| rank 32, 12 ep, after book 4 | +1.7 | -10.5 | -2.4 | 11.2 |
| rank 32, 6 ep, after book 4 | -2.8 | -2.7 | -1.3 | 6.8 |
| round 22, after book 4 | **+0.2** | **-4.2** | **-1.0** | **5.2** |
| round 23 = round 22 + 320 self-written math problems in the protected set (576 rows, tau 0.90) | -4.3 | -3.2 | -1.0 | 8.5 |
Round 23 changed nothing else and left L1-L4 unchanged (knowledge 149 x4, traps 0/1/0/0, twins 0 x4); GSM8K stayed at
-3 (three 6-epoch runs: -2.7 / -4.2 / -3.2) and IFEval dropped 4.3, so enlarging the protected set does not buy GSM8K.
| round 24 = round 22 with book 1 at 4 epochs | +0.9 | -9.5 | -0.9 | 10.4 |
Round 24 (4 epochs) broke L1 (book 1 made up 21/60 right after training, and every later book 18-20/60: the first
book fixes the premise-checking behaviour for all later books) and GSM8K fell to -9.5, so the epochs-to-damage relation
is not monotonic; 64 sampled GSM8K answers were genuine arithmetic/logic errors, not extraction artifacts. Six epochs
is the working point for the first book: fewer loses L1 without helping L5, more costs L5 and L1.
| round 25 = round 22 with seed 1 (the second run of L6) | -4.4 | -7.0 | -1.3 | 12.7 |

## Two seeds of the same configuration (rounds 22 and 25): L6 and the real run-to-run variance
| after book 4 | seed 0 | seed 1 | difference |
|---|---|---|---|
| knowledge b1/b2/b3/b4 | 150 / 150 / 150 / 150 | 149 / 150 / 149 / 146 | 1 / 0 / 1 / 4 (L6 line: <= 9) |
| retention b1/b2/b3 | 1.000 / 1.000 / 1.000 | 0.993 / 1.000 / 0.993 | |
| traps made up /60 | 0 / 1 / 1 / 0 | 2 / 2 / 0 / **5** | book 4 fails <= 2 in the second run |
| twin false refusals /30 | 0 / 0 / 1 / 1 | 0 / 0 / 1 / 0 | |
| general-text NLL / chat NLL | 2.305 / 0.650 | 2.309 / 0.654 | the same |
| L5 IFEval / GSM8K / MMLU, sum of drops | +0.2 / -4.2 / -1.0, 5.2 | -4.4 / -7.0 / -1.3, 12.7 | 4.6 / 2.8 / 0.3 |
Knowledge is stable across seeds (L6 passes). Traps are not: with ~60 traps per book and a true made-up rate around
2%, passing <= 2 on all four books in both runs is a coin flip (the client's page gives 36% at 2%). L5 varies with the
training seed far more than its evaluation noise: +0.2 vs -4.4 on IFEval and -4.2 vs -7.0 on GSM8K for identical data
and hyper-parameters, while the two NLL proxies are unchanged. Single-run L5 numbers therefore cannot rank
configurations at the 1-3 point level, and the expected L5 loss of the best configuration is about 8-9 points, not 5.
Without any general protection a small-book run lost 18.7 points in total. The damage is in multi-step arithmetic
(GSM8K) and, less, in knowledge (MMLU); instruction following does not drop. It scales with the training volume of the
first book (12 -> 6 epochs: -10.5 -> -2.7/-4.2 on GSM8K) and with the rank. IFEval moves +-3 and GSM8K +-1.5 between
runs of the same kind (541 / 1,319 items), so single runs cannot resolve the client's 1-point line; GSM8K is the one
quantitative line still failed (-2.7 and -4.2 in the two 6-epoch runs).

## Reproducibility, cost, engineering
- Merged checkpoints rebuilt from the saved LoRA factors reproduce the training-time answers (157-160/160 identical
  first five words). The same weights on TPU vs A100 flip 1-2 of 180 trap/twin outcomes (bf16 kernels).
- Training speed, 1.5B, batch 32: TPU v5e-1 0.366 s/step (book 1: 12 epochs 101 min, 6 epochs 50 min); A100 40 GB
  0.41 s/step with this JAX code. 7B on A100 80 GB (JAX, earlier round): 1.00 s/step. Estimate for the client's A40
  (48 GB, ~0.45x A100): book 1 at 6 epochs ~5-6 GPU-hours incl. subspaces and self-tests (12 epochs: 9-12, over the
  8-hour line); books 2-4 ~3-4.5 each. 48 GB memory at book 4 for the 7B is a risk (bases ~10-15 GB in fp32).
- The run is restartable: per-epoch checkpoints (LoRA, optimizer, shuffle rng, review boxes) and per-book LoRA files
  synced to Drive; resume is bit-exact (verified on 0.5B smoke runs and on the real book 1).

## Not measured / open
- The 7B itself, end to end; the client's books and exam; two runs with different seeds (L6, and the GSM8K noise).
- An honesty test the user asked for: hold out one or two (asked, premise) pairs from training entirely and score
  traps/twins on them (currently every pair is trained, so only wordings are held out).
- Why GSM8K still drops with 80 math word problems in the protected set (round 22: tau 0.90): the next step is a
  math-heavy self-instruct set.

## Part 3: the 7B on book 1 (round 26, 2026-10-27)

Setup: Qwen2.5-7B-Instruct, the round-22 configuration unchanged (data_v17, LoRA rank 32, 6 epochs on book 1,
tau_chat 0.90 with the 7B writing its own 320 self-instruct rows, seed 0), on a Colab G4 (RTX PRO 6000 Blackwell,
96 GB). Book 1 only (2,176 units, 44,377 rows); books 2-4 not run.

Cost and memory (L7 evidence): 0.81 s/step, 6 epochs = 6,793 s (1 h 53 min) of training; device peak 43.8 GiB during
training and 50.3 GiB during the self-test (generation batch 48); book-1 protected subspace after training: mean 1,250
dims per module, max 2,999 (1.5B: 806 / 2,330). Scaled to an A40/A6000 (about half the bandwidth, a third of the
compute), book 1 is 4-5.5 GPU-hours, inside the 8-hour line; the 48 GB card does not hold the current peaks, so bf16
bases, micro-batch 4 and a smaller generation batch are required before a 7B run on the client's hardware.

Scores after book 1 (same exams as the proxy):

| after book 1 | 1.5B (round 22) | 7B (round 26) |
|---|---|---|
| main exam direct / indirect (/75 each) | 75 / 75 | 75 / 60 |
| second exam direct / indirect | 71 / 75 | 75 / 75 |
| traps made up, main exam (/60) | 0 | 20 (all founder_swap; all 20 state the true year without denying the false founder) |
| traps made up, second exam (/120) | 0 | 0 |
| twin false refusals, main / second | 0 / - | 0 / 0 (one twin answered with a wrong year) |
| general-text NLL, base -> after | 1.893 -> 2.240 (+0.35) | 1.599 -> 2.016 (+0.42) |
| self-instruct NLL, base -> after | 0.570 -> 0.623 (+0.05) | 0.395 -> 0.475 (+0.08) |

Both losses are the "founder" relation in exam wordings that training never uses (exam wordings are held out by
design): "В каком году [false founder] основал город X?" is answered with the true year and no denial (the shortcut the
1.5B had at rank 16 and lost at rank 32), while the same (year asked, founder premise) pair in the second exam's
wordings is denied 20/20; and the reverse lookup "Основателем какого города был X?" (training has one reverse wording,
"Какой город основал X?", 435 rows) gets 1/16, the model naming a wrong city and then denying the true founder.
Conversely the 7B answers "Кому принадлежит основание города X?" 75/75 without the templates the 1.5B needed. The 7B
generalises to different wordings than the 1.5B, not to more of them; for an unknown client exam this is a robustness
problem, and the remedy on the table is the same as round 22: teacher-written wording diversity for reverse lookups
and for year questions that name the founder. The two L5 proxies drop more on the 7B than on the 1.5B, so the earlier
"7B is damaged less" observation (rank 16, small books) did not reproduce on the large book; actual L5 on the 7B is
not measured yet.

L5 on the 7B after book 1 (same harness and commit as Part 2, bf16 greedy, MMLU batch 8 on the 96 GB card):

| 7B, L5 | base | after book 1 | change | 1.5B after book 4 (round 22) |
|---|---|---|---|---|
| IFEval prompt-level strict | 70.79 | 72.27 | +1.48 | +0.2 |
| GSM8K flexible-extract | 81.50 | 80.59 | -0.91 | -4.2 |
| GSM8K strict-match | 75.51 | 79.45 | +3.94 | (format-driven on the 1.5B) |
| MMLU | 74.24 | 73.29 | -0.95 | -1.0 |
| sum of drops (flexible-extract) | | | 1.86 | 5.2 |

Each task is inside its 1.0-point line; the sum of drops is 1.86 against 1.5 with GSM8K read as flexible-extract
(0.95, passing, if read as strict-match, where the trained model gains). Standard errors are 1.1 (GSM8K) and 0.35
(MMLU), so both drops sit on the line within noise. One seed, book 1 only; on the proxy the seed moved L5 by 4-7
points, and the 7B's seed variance is not measured. The two NLL proxies moved the other way (the 7B lost more NLL
than the 1.5B), which again limits what they can say about benchmark scores. The merged checkpoint reproduced the
training-time answers 40/40.

Round 27 (same 7B run, training wordings only): the two losses above were data coverage, not capacity. The reverse
founder lookup had two training wordings (plus the base one) and the year-asked / founder-named family had no short
single-clause wording with the founder as subject. Teacher-written templates for both (Qwen2.5-7B writes, Vikhr-Nemo-12B
judges, rule checks, exam wordings and near-copies at Jaccard >= 0.75 excluded, final selection by hand with every
rejection reason recorded) took inverse/founder from 2 to 26 templates and premise/founded|founder from 21 to 60;
books, facts and both exams are byte-identical, the training set grew by one row. Result after book 1 on the 7B:
main exam 149/150 (indirect 74/75), traps made up 0/60, twins 30/30 and 60/60, second exam 150/150 and 0/120; the two
NLL proxies also improved (general-text +0.31 instead of +0.42, self-instruct +0.067 instead of +0.081). Same
training time and memory as round 26.

L5 of the round-27 model (book 1, data_v18), same harness, base reused: IFEval -0.74, GSM8K flexible-extract -1.29
(strict-match +0.30), MMLU -1.03; sum of drops 3.06 (flexible) / 1.77 (strict), against 1.86 / 0.95 for round 26.
Only the training wordings changed, yet the three scores moved by -2.2, -0.4 and -0.1 points while both NLL proxies
improved. Two runs of the 7B after book 1 therefore put L5 at the line rather than inside it, and the NLL proxies do
not predict the benchmark scores.

Books 2-4 on the 7B (round 27 continued, same run, seed 0): after book 4 every main-exam score is 150/150, traps made
up 0/0/0/0 of 60 (0/120 x4 on the second exam), twin false refusals 0/0/0/0 of 30 (0/0/1/0 of 60 on the second exam),
retention of books 1-3 1.000/1.000/1.000. Training 40/26/33 min for books 2-4 (2.05x the 1.5B); device peak 52.9 GiB
during book 4 (fp32 bases, micro-batch 8), 50.3 GiB in the self-test. Merged checkpoint vs training-time answers:
160/160 identical. L5 after book 4: IFEval +0.55, GSM8K flexible-extract -0.68 (strict-match +3.71), MMLU -1.05; sum of
drops 1.73 (flexible) / 1.05 (strict). Across the three 7B measurements (two after book 1, one after book 4) the sum is
1.7-3.1 flexible / 0.95-1.8 strict: at the line, not inside it; MMLU is the steady loss (-0.95 to -1.05), the other two
move by 1-3 points between runs. Books 2-4 add almost nothing to L5 (NLL proxies +0.04 / +0.03 from book 1 to book 4).
Open: a second seed (L6 and the L1/L5 variance), the 48 GB memory fit (bf16 bases, micro-batch 4 not yet implemented),
the client's own books and exam, the held-out-pair honesty test.

Memory fit for a 48 GB card (round 28, 2026-10-27). Changes: protection bases stored in fp16 on the device (half the
size; leakage |A Q|/|A| ~1e-4 instead of ~3e-7, set by the bases' own rounding), LoRA deltas merged on the host and
device weights replaced one kind at a time, the projection applied layer by layer (the batched einsum made XLA copy the
whole 7 GiB down_proj basis), cuBLAS instead of Triton GEMMs, micro-batch 2 for the last book, generation batch 16.
With JAX capped at 39.9 GiB on the 96 GB card (the share a 48 GB card would give it), books 1-3 were loaded from the
round-27 factors and book 4 was trained: device peak 32.6 GiB, and the scores are identical to the fp32 run (four books
150/150, traps 0/60 and 0/120 each, twin false refusals 0, retention 1.000 x3). Cost of micro-batch 2: book 4 took 83 min
instead of 33; book 1 does not need it (its bases are small), so the packaged run.sh picks the micro-batch per book from
the stacked basis size. Four earlier attempts went out of memory at four different places (device-side merge, its
copies in the branch code, the basis copy in the projection, the training step with a fragmented pool); a 4-minute probe
with zero bases of the book-4 shapes (MEMTEST) found the working configuration faster than full runs did.
L5 of that memory-fitted model (fp16 bases, micro-batch 2 for book 4; client metrics IFEval strict, GSM8K strict,
MMLU): -2.77 / +0.99 / -1.44, sum of drops 4.21, against +0.55 / +3.71 / -1.05 (1.05) for the same books trained with
fp32 bases and micro-batch 8. Knowledge, traps, twins and retention are identical between the two; only L5 moved. The
micro-batch size changes the training objective (the step loss averages tokens within a micro-batch and then averages
micro-batches, so a smaller micro-batch up-weights short rows), so the memory-fitted configuration is not the validated
one; gradient checkpointing at micro-batch 8 is the candidate that keeps the objective and the memory budget.
Same books trained with fp16 bases, gradient checkpointing and micro-batch 8 (the configuration that fits a 48 GB card
with the validated objective): L1-L4 identical, L5 IFEval -2.22 / GSM8K strict +0.23 / MMLU -1.23, sum of drops 3.45
(fp32 run: 1.05). The three book-4 trainings differ only in numerics (precision of the bases, checkpointing, GEMM
library); the spread of their L5 (1.05 to 4.21) is the run-to-run noise of this measurement on the 7B and is larger
than the client's margins.

## Part 4: rows written by the base model from arbitrary fact units (round 29, 2026-10-27)

The client's books are arbitrary Russian fact units; every row so far came from templates over six synthetic relations.
A generic generator (code/generic.py) lets the base model itself write, per unit: subject, key detail, a direct and an
inverse question; three paraphrases; two single-detail falsifications; and premise questions pairing the unit with
other units of the same subject (true twin: the question states the partner's fact; false twin: it states a
falsified version; answers follow the same formats as the template rows). Three iterations on synthetic book 2 (145
units, book 1 loaded from the template-trained run, the real exam):
- filters too strict (inverse question required to state the detail): 45/145 units parsed;
- relaxed, but partners drawn at random across subjects when a subject had none: knowledge 149, traps made up 8/60
  twice (doubling false rows did not help) - the model learned to attribute a false premise to another subject;
- same-subject partners only, subjects grouped by a normalised key: knowledge 149/150, traps 0/60, twins 30/30, second
  exam 150 and 1/120, book 1 kept (149, retention 0.993). 1,930 rows for 145 units, written in 93 s on the 96 GB card.
Not yet run with generic rows: the 2,176-unit book. Generation time scales linearly (about 1 s per unit at batch 48).
