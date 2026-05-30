#!/usr/bin/env bash
# Smoke-test an existing model on AIME with the native CodeTool.
#
# Typical usage:
#   MODEL_PATH=/path/to/model AIME_PARQUET=/path/to/aime.parquet bash examples/code_tool/run_aime_code_tool_eval.sh
#
# Optional knobs:
#   CONDA_ENV=zhouyz
#   LIMIT=8
#   INFER_BACKEND=vllm
#   TOOL_FORMAT=qwen3_coder
#   NGPUS_PER_NODE=1
#   ROLLOUT_TP=1

set -xeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"

CONDA_ENV="${CONDA_ENV:-zhouyz}"
CONDA_RUN=(conda run --no-capture-output -n "${CONDA_ENV}")

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
INFER_BACKEND="${INFER_BACKEND:-vllm}"
TOOL_FORMAT="${TOOL_FORMAT:-qwen3_coder}"
LIMIT="${LIMIT:-8}"
WORK_DIR="${WORK_DIR:-${PROJECT_DIR}/outputs/aime_code_tool_smoke}"
DATA_DIR="${DATA_DIR:-${WORK_DIR}/data}"
TOOL_CONFIG_PATH="${TOOL_CONFIG_PATH:-${WORK_DIR}/code_tool_config.yaml}"
RAW_AIME_DIR="${RAW_AIME_DIR:-${DATA_DIR}/raw_aime2024}"
PREPARED_AIME_PARQUET="${PREPARED_AIME_PARQUET:-${DATA_DIR}/aime_code_tool.parquet}"

NGPUS_PER_NODE="${NGPUS_PER_NODE:-1}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_N="${ROLLOUT_N:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${LIMIT}}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${LIMIT}}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
MAX_TOOL_RESPONSE_LENGTH="${MAX_TOOL_RESPONSE_LENGTH:-2048}"
CODE_TIMEOUT="${CODE_TIMEOUT:-10}"

mkdir -p "${WORK_DIR}" "${DATA_DIR}" "${RAW_AIME_DIR}"

cat > "${TOOL_CONFIG_PATH}" <<EOF
tools:
  - class_name: "verl.tools.code_tool.CodeTool"
    config:
      type: native
      default_timeout: ${CODE_TIMEOUT}
      max_timeout: ${CODE_TIMEOUT}
      max_output_chars: ${MAX_TOOL_RESPONSE_LENGTH}
      default_language: python
      allowed_languages:
        - python
        - py
EOF

if [[ -n "${AIME_PARQUET:-}" ]]; then
    SOURCE_AIME_PARQUET="${AIME_PARQUET}"
else
    if [[ ! -f "${RAW_AIME_DIR}/train.parquet" ]]; then
        "${CONDA_RUN[@]}" python examples/data_preprocess/aime2024_multiturn_w_tool.py \
            --local_save_dir "${RAW_AIME_DIR}"
    fi
    SOURCE_AIME_PARQUET="${RAW_AIME_DIR}/train.parquet"
fi

"${CONDA_RUN[@]}" python - <<PY
from pathlib import Path

import pandas as pd

source = Path("${SOURCE_AIME_PARQUET}").expanduser()
target = Path("${PREPARED_AIME_PARQUET}").expanduser()
limit = int("${LIMIT}")

df = pd.read_parquet(source)
if limit > 0:
    df = df.head(limit).copy()

df["agent_name"] = "tool_agent"

def patch_extra_info(row):
    extra = row.get("extra_info")
    if not isinstance(extra, dict):
        extra = {}
    else:
        extra = dict(extra)

    reward_model = row.get("reward_model")
    ground_truth = None
    if isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth")

    extra["need_tools_kwargs"] = True
    extra["tools_kwargs"] = {
        "code_interpreter": {
            "create_kwargs": {
                "ground_truth": ground_truth,
            },
        },
    }
    return extra

df["extra_info"] = df.apply(patch_extra_info, axis=1)
target.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(target)
print(f"Wrote {len(df)} AIME rows with tool_agent to {target}")
PY

"${CONDA_RUN[@]}" python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="['${PREPARED_AIME_PARQUET}']" \
    data.val_files="['${PREPARED_AIME_PARQUET}']" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.name="${INFER_BACKEND}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG_PATH}" \
    actor_rollout_ref.rollout.multi_turn.format="${TOOL_FORMAT}" \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length="${MAX_TOOL_RESPONSE_LENGTH}" \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    trainer.project_name=aime_code_tool_smoke \
    trainer.experiment_name="aime_code_tool_${INFER_BACKEND}_${TOOL_FORMAT}" \
    trainer.logger='["console"]' \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    "$@"
