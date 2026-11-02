# continual-book-lora

Continual learning of four Russian fact books into Qwen2.5-7B-Instruct for an Upwork client (fixed acceptance lines, offline package). The project is paused; **read `HANDOFF.md` first** — status, results, open problems, next steps, how to run an experiment. Per-run log: `tpu/JAX_PLAN.md`.

## Rules

- The user writes in Chinese; answer in Chinese.
- Before opening any Colab GPU/TPU (or a RunPod pod), state the expected duration and cost and wait for the user's go. Pull results first, then stop the session at once; never leave an idle machine. Keep long runs alive with `colab exec -s NAME -f tools/ping.py` every 8 minutes.
- The user optimises for time, not cost: prefer the faster hardware and parallel runs.
- Report traps and twins as higher-is-better under strict scoring: traps caught x/60 (line ≥ 58), twins not refused x/30 (line ≥ 28). Re-answer with `lab/hf_trap_answers.py` (transformers, fp16, greedy, 128 tokens) and score with `lab/score_strict.py`.
- Judge any change with two seeds; do not draw conclusions from one L5 point (IFEval/GSM8K single points are mostly noise, see HANDOFF.md §4.3).
- Check every new idea against the client's dead-ends list (HANDOFF.md §7). No Claude-written training data: all training rows come from the base model.
- Never commit or print secrets (`/home/ubuntu/.config/rclone/rclone.conf`), never commit client material (`client/`), never upload client books or exams anywhere.
- Upwork actions (bids, Connects, messages, proposal edits) are the user's decision; show the full text first.
- Commit as `pingkui <pingkui@users.noreply.github.com>`; write each run's result into `tpu/JAX_PLAN.md` before committing.
- Experiment bundles for the Colab cells: `bash tools/build_bundles.sh OUT_DIR`.
