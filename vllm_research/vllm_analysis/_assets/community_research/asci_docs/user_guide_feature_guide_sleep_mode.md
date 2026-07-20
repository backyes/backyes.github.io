Sleep Mode Guide

Overview

Sleep Mode is an API designed to offload model weights and discard KV cache from NPU memory. This functionality is essential for reinforcement learning (RL) post-training workloads, particularly in online algorithms such as PPO, GRPO, or DPO. During training, the policy model typically performs autoregressive generation using inference engines like vLLM, followed by forward and backward passes for optimization.

Since the generation and training phases may employ different model parallelism strategies, it becomes crucial to free KV cache and even offload model parameters stored within vLLM during training. This ensures efficient memory utilization and avoids resource contention on the NPU.

Getting started

With enable_sleep_mode=True, the way we manage memory (malloc, free) in vLLM is under a specific memory pool. During model loading and KV cache initialization, we tag the memory as a map: {"weight": data, "kv_cache": data}.

The engine (v0/v1) supports two sleep levels to manage memory during idle periods:

- Level 1 Sleep
    - Action: Offloads model weights and discards the KV cache.
    - Memory: Model weights are moved to CPU memory; KV cache is forgotten.
    - Use Case: Suitable when reusing the same model later.
    - Note: Ensure sufficient CPU memory is available to hold the model weights.

- Level 2 Sleep
    - Action: Discards both model weights and KV cache.
    - Memory: The content of both the model weights and KV cache is forgotten.
    - Use Case: Ideal when switching to a different model or updating the current one.

Since this feature uses the low-level API AscendCL, in order to use sleep mode, you should follow the installation guide and build from source. If you are using < v0.12.0rc1, remember to set export COMPILE_CUSTOM_KERNELS=1.

Optional extra cleanup

By default, sleep mode only releases memory managed by the sleep-mode allocator. For RL workloads that need to return more NPU memory to the trainer, vLLM Ascend also provides an optional extra cleanup path:



For online serving, pass the same option through --additional-config:



When enable_sleep_mode_extra_cleanup is enabled, sleep() additionally:

- clears ACL graph attention workspaces and invalidates captured ACL graph caches when ACL graph is enabled;
- resets the model runner graph manager so ACL graphs can be captured again after wakeup;
- waits for pending pipeline-parallel send work, synchronizes the NPU, and destroys HCCL process groups.

During wake_up(), vLLM Ascend restores the HCCL process groups, refreshes MoE dispatcher HCCL metadata, restores sleep-mode allocator memory, and recaptures ACL graphs when needed.

!!! note

    Extra cleanup trades lower sleep-time NPU memory usage for longer wakeup latency. In particular, if ACL graph is enabled, wake_up() must call capture_model() again after the model state has been restored. Keep enable_sleep_mode_extra_cleanup disabled when lower wakeup latency is more important than releasing HCCL and ACL graph workspace memory.

For level 2 sleep, wakeup can be split into two phases:



With extra cleanup enabled, ACL graphs are recaptured only when tags is None or contains "kv_cache". This avoids recapturing graphs before externally reloaded weights and KV-cache state are ready.

Expert weight layout restoration

For dense models, wake_up() simply restores the model weights to NPU memory; the tensor layout is unchanged.

For unquantized MoE models (quant_config is None), the fused expert weights are stored in a transposed layout for NPU matmul efficiency. This layout is produced once at model load time by process_weights_after_loading(): after the weights are loaded, the method transposes the second and third dimensions (transpose(1, 2)) of w13_weight and w2_weight to convert the standard checkpoint layout into the format required by the torch_npu.npu_grouped_matmul operator.

After the sleep-mode allocator restores the original (untransposed) memory, wake_up() re-applies the same transpose to the affected expert weights when the "weights" tag is being restored:

- w13_weight (gate/up projection): transposed back to the runtime layout when its second dimension matches hidden_size;
- w2_weight (down projection): transposed back to the runtime layout when its third dimension matches hidden_size.

This step is skipped entirely for dense models (which have no expert weights) and for quantized models (whose weights are handled by the quantization method).

Prepare Model Weights

Use the Qwen2.5-0.5B-Instruct model weights. With VLLM_USE_MODELSCOPE=True, the model will be downloaded automatically from ModelScope.



Usage

The following is a simple example of how to use sleep mode.

- Offline inference:

    

- Online serving:
    !!! note

            Considering there may be a risk of malicious access, please make sure you are under a dev-mode, and explicitly specify the dev environment VLLM_SERVER_DEV_MODE to expose these endpoints (sleep/wake up).

    
