#!/usr/bin/env bash
# Smoke-test an existing model on AIME with the native CodeTool.
#
# Typical usage:
#   MODEL_PATH=/path/to/model AIME_PARQUET=/path/to/aime.parquet bash examples/code_tool/run_aime_code_tool_eval.sh
#
# Optional knobs:
#   LIMIT=8
#   INFER_BACKEND=vllm
#   TOOL_FORMAT=hermes
#   NGPUS_PER_NODE=8
#   ROLLOUT_TP=8

set -xeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"

CONDA_ENV="${CONDA_ENV:-va}"
CONDA_RUN=(conda run --no-capture-output -n "${CONDA_ENV}")

MODEL_PATH="${MODEL_PATH:-/workspace/hf/Qwen/QwQ-32B}"
INFER_BACKEND="${INFER_BACKEND:-vllm}"
TOOL_FORMAT="${TOOL_FORMAT:-hermes}"
PROJECT_NAME="${PROJECT_NAME:-aime_code_tool_smoke}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-aime_code_tool_qwq}"
LIMIT="${LIMIT:-8}"
WORK_DIR="${WORK_DIR:-${PROJECT_DIR}/outputs/aime_code_tool_smoke}"
DATA_DIR="${DATA_DIR:-${WORK_DIR}/data}"
TOOL_CONFIG_PATH="${TOOL_CONFIG_PATH:-${WORK_DIR}/code_tool_config.yaml}"
RAW_AIME_DIR="${RAW_AIME_DIR:-${DATA_DIR}/raw_aime2024}"
PREPARED_AIME_PARQUET="${PREPARED_AIME_PARQUET:-${DATA_DIR}/aime_code_tool.parquet}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-${WORK_DIR}/validation_generations}"
SYSTEM_PROMPT_PATH="${SYSTEM_PROMPT_PATH:-${PROJECT_DIR}/examples/code_tool/system_prompt_qwq.txt}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${WORK_DIR}/tensorboard/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
TRAINER_LOGGER="${TRAINER_LOGGER:-[console,tensorboard]}"

NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
ROLLOUT_TP="${ROLLOUT_TP:-${NGPUS_PER_NODE}}"
ROLLOUT_N="${ROLLOUT_N:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${LIMIT}}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${LIMIT}}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-3072}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-3072}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-6144}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-False}"
ROLLOUT_FREE_CACHE_ENGINE="${ROLLOUT_FREE_CACHE_ENGINE:-True}"
ROLLOUT_ENABLE_CHUNKED_PREFILL="${ROLLOUT_ENABLE_CHUNKED_PREFILL:-True}"
ROLLOUT_ENABLE_PREFIX_CACHING="${ROLLOUT_ENABLE_PREFIX_CACHING:-False}"
FSDP_SIZE="${FSDP_SIZE:-${NGPUS_PER_NODE}}"
SP_SIZE="${SP_SIZE:-1}"
MAX_TOOL_RESPONSE_LENGTH="${MAX_TOOL_RESPONSE_LENGTH:-2048}"
CODE_TIMEOUT="${CODE_TIMEOUT:-10}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${WORK_DIR}" "${DATA_DIR}" "${RAW_AIME_DIR}"
export TENSORBOARD_DIR PYTORCH_CUDA_ALLOC_CONF

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

"${CONDA_RUN[@]}" python examples/code_tool/prepare_aime_code_tool_data.py \
    --source "${SOURCE_AIME_PARQUET}" \
    --target "${PREPARED_AIME_PARQUET}" \
    --system-prompt "${SYSTEM_PROMPT_PATH}"

"${CONDA_RUN[@]}" python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="['${PREPARED_AIME_PARQUET}']" \
    data.val_files="['${PREPARED_AIME_PARQUET}']" \
    data.train_max_samples="${LIMIT}" \
    data.val_max_samples="${LIMIT}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size="${FSDP_SIZE}" \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.actor.fsdp_config.offload_policy=True \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size="${SP_SIZE}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.ref.fsdp_config.offload_policy=True \
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size="${SP_SIZE}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.name="${INFER_BACKEND}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}" \
    actor_rollout_ref.rollout.free_cache_engine="${ROLLOUT_FREE_CACHE_ENGINE}" \
    actor_rollout_ref.rollout.enable_chunked_prefill="${ROLLOUT_ENABLE_CHUNKED_PREFILL}" \
    actor_rollout_ref.rollout.enable_prefix_caching="${ROLLOUT_ENABLE_PREFIX_CACHING}" \
    actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=6144 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG_PATH}" \
    actor_rollout_ref.rollout.multi_turn.format="${TOOL_FORMAT}" \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length="${MAX_TOOL_RESPONSE_LENGTH}" \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger="${TRAINER_LOGGER}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.validation_data_dir="${VALIDATION_DATA_DIR}" \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    "$@"
