Batch Invariance

!!! note

    Batch invariance is currently in beta. Some features are still under active development.
    Track progress and planned improvements at <https://github.com/vllm-project/vllm-ascend/issues/5487>

!!! note

    To install the batch invariance custom operator library, set VLLM_BATCH_INVARIANT=1 before building vllm-ascend.
    For installation instructions, see <https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/installation.md#set-up-using-python>

This document shows how to enable batch invariance in vLLM-Ascend. Batch invariance ensures that the output of a model is deterministic and independent of the batch size or the order of requests in a batch.

Motivation

Batch invariance is crucial for several use cases:

- Framework debugging: Deterministic outputs make it easier to debug issues in the inference framework, as the same input will always produce the same output regardless of batching.
- Model debugging: Helps identify issues in model implementations by ensuring consistent behavior across different batch configurations.
- Reinforcement Learning (RL): RL training often requires deterministic rollouts for reproducibility and stable training.
- Large-scale inference systems: Systems that use vLLM as a component benefit from deterministic behavior for testing, validation, and consistency guarantees.

Hardware Requirements

Batch invariance currently requires Ascend Atlas A2 and A3 inference products NPUs.
We will support Ascend 950 Products and other NPUs in the future.

Software Requirements

Batch invariance requires a custom operator library for Atlas A2 and A3 inference products, and users need to set VLLM_BATCH_INVARIANT=1 before building vllm-ascend to install the batch invariance custom operator library during the installation process.

Enabling Batch Invariance

Batch invariance can be enabled by setting the VLLM_BATCH_INVARIANT environment variable to 1:



Online Inference (Server Mode)

To start a vLLM server with batch invariance enabled:



Then use the OpenAI-compatible client:



Offline Inference

For offline batch inference with batch invariance:



Tested Models

Batch invariance has been tested and verified on the following models:

- Qwen3 (Dense): Qwen/Qwen3-1.7B, Qwen/Qwen3-8B
- Qwen3 (MoE): Qwen/Qwen3-30B-A3B, Qwen/Qwen3-235B-A22B

Other models may also work, but these have been explicitly validated. If you encounter issues with a specific model, please report them on the GitHub issue tracker.

Implementation Details

When batch invariance is enabled, vLLM:

1. Uses deterministic kernel implementations for attention and other operations
2. Ensures consistent numerical behavior across different batch sizes
3. Disables certain optimizations that may introduce non-determinism

!!! note

    The batch invariance attention operators currently do not support
    FULL'、'FULL_DECODE_ONLY cudagraph mode.

!!! note

    Enabling batch invariance may impact performance compared to the default non-deterministic mode. This trade-off is intentional to guarantee reproducibility.

Future Improvements

The batch invariance feature is under active development. Planned improvements include:

- Support for additional NPUs series
- Support FULL'、'FULL_DECODE_ONLY cudagraph mode with batch invariance attention operators
- Expanded model coverage
- Performance optimizations
- Additional testing and validation

For the latest status and to contribute ideas, see the tracking issue.
