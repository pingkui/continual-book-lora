#!/bin/bash
# L5 on a Colab A100 (Qwen2.5-1.5B proxy): IFEval (prompt-level strict), GSM8K (5-shot strict) and full MMLU (5-shot),
# with lm-evaluation-harness at the client's pinned commit, bf16 greedy, for the clean base and the round-14 seed-0 model
# rebuilt from its LoRA factors (export_lora.py), whose answers are first checked against the TPU run (check_export.py).
#   bash run_l5.sh prep   install, rebuild, check, and a timing trial (LIMIT items per task, both models)
#   bash run_l5.sh trial  only the timing trial (after prep)
#   bash run_l5.sh full   the full tasks for both models
#   bash run_l5.sh book NAME LORA...   the full tasks for the model merged from the given factors (L5 after book k)
set -u
L=/content/l5
cd $L/code
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
BASE=${BASE:-Qwen/Qwen2.5-1.5B-Instruct}   # round 26: BASE=Qwen/Qwen2.5-7B-Instruct for the 7B
TRAINED=$L/trained
RUN=${RUN:-A4_gen70n}                          # run name in the results file (check_export.py)
RES=${RES:-$L/results_v14s0save.json}          # that run's results file
run_tasks() {  # name model limit
  local name=$1 model=$2 lim=${3:-}
  local extra=""; [ -n "$lim" ] && extra="--limit $lim"
  # task:shots:batch[:chat]. Explicit batches: "auto" picked 1 for ifeval's 1280-token generations, and MMLU 5-shot
  # contexts (up to ~3000 tokens x 152k-word logits) ran out of the 40 GB A100's memory at batch 64 and 16. IFEval with the chat template: the client's floor (70.35 for the 7B Instruct
  # model) is only reached with it; GSM8K and MMLU with the harness defaults.
  for spec in "ifeval:0:${L5_BS:-64}:chat" "gsm8k:5:${L5_BS:-64}" "mmlu:5:${MMLU_BS:-4}"; do
    IFS=: read -r t k bs chat <<<"$spec"
    case " ${L5_TASKS:-ifeval gsm8k mmlu} " in *" $t "*) ;; *) continue ;; esac   # L5_TASKS: a subset of the tasks
    if [ -n "$(find $L/out/$name/$t -name 'results_*.json' 2>/dev/null)" ]; then echo "$name $t: done before"; continue; fi
    local ct=""; [ "${chat:-}" = chat ] && ct="--apply_chat_template"
    s=$(date +%s)
    lm_eval --model hf --model_args pretrained=$model,dtype=${L5_DTYPE:-bfloat16} --tasks $t --num_fewshot $k --batch_size $bs $ct \
      --output_path $L/out/$name/$t $extra > $L/out/$name.$t.log 2>&1 || { echo "LM_EVAL FAILED $name $t"; tail -20 $L/out/$name.$t.log; return 1; }
    echo "$name $t: $(( $(date +%s) - s ))s"
    grep -E "prompt_level_strict_acc|strict-match|^\|mmlu " $L/out/$name.$t.log | head -4
  done
}
mkdir -p $L/out
if [ "$1" = prep ]; then
  pip uninstall -y -q torchao 2>/dev/null
  pip install -q "lm_eval[ifeval] @ git+https://github.com/EleutherAI/lm-evaluation-harness.git@ddd67220" \
    || { echo "PINNED INSTALL FAILED"; exit 1; }
  python -c "import lm_eval, importlib.metadata as m; print('lm_eval', m.version('lm_eval'))"
  # LORAS: the saved factors to merge, in book order (default: the round-14 books 2-4; a book-1-first run lists book1 too)
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 python export_lora.py $BASE $TRAINED \
    ${LORAS:-$L/lora/book2.npz $L/lora/book3.npz $L/lora/book4.npz} || { echo "EXPORT FAILED"; exit 1; }
  python check_export.py $TRAINED $RES $RUN ${BRANCH:-sr12} $L/data 40   # BRANCH=- : the main run (book-1-only runs)
  echo "L5 PREP DONE"
elif [ "$1" = book ]; then
  # bash run_l5.sh book NAME LORA...: merge the given factors (books 1..k) into $L/NAME, run the full tasks under NAME,
  # then delete the merged model (~15 GB). For L5 after each book; no export check (the results file has no per-book model).
  name=$2; shift 2
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 python export_lora.py $BASE $L/$name "$@" || { echo "EXPORT FAILED $name"; exit 1; }
  run_tasks $name $L/$name && echo "L5 $name DONE"
  rm -rf $L/$name
elif [ "$1" = model ]; then
  # bash run_l5.sh model NAME DIR: the full tasks for an existing model dir or hub name (noise tests; L5_DTYPE, L5_BS)
  run_tasks $2 $3 && echo "L5 $2 DONE"
elif [ "$1" = trial ]; then
  run_tasks base_trial $BASE ${LIMIT:-40} && run_tasks trained_trial $TRAINED ${LIMIT:-40} && echo "L5 TRIAL DONE"
else
  run_tasks base $BASE && run_tasks trained $TRAINED && echo "L5 FULL DONE"
fi
