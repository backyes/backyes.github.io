Performance Benchmark

This document details the benchmark methodology for vllm-ascend, aimed at evaluating the performance under a variety of workloads. To maintain alignment with vLLM, we use the benchmark script provided by the vllm project.

Benchmark Coverage: We measure offline E2E latency and throughput, and fixed-QPS online serving benchmarks. For more details, see vllm-ascend benchmark scripts.

Legend Description:

- ✅ = Supported
- 🟡 = Partial / Work in progress
- 🚧 = Under development
  
1. Run docker container



2. Install dependencies



3. Run basic benchmarks

This section introduces how to perform performance testing using the benchmark suite built into VLLM.

3.1 Dataset

VLLM supports a variety of datasets.

<style>
th {
  min-width: 0 !important;
}
</style>

| Dataset | Online | Offline | Data Path |
|---------|--------|---------|-----------|
| ShareGPT | ✅ | ✅ | wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json |
| ShareGPT4V (Image) | ✅ | ✅ | wget https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/resolve/main/sharegpt4v_instruct_gpt4-vision_cap100k.json<br>Note that the images need to be downloaded separately. For example, to download COCO's 2017 Train images:<br>wget http://images.cocodataset.org/zips/train2017.zip |
| ShareGPT4Video (Video) | ✅ | ✅ | git clone https://huggingface.co/datasets/ShareGPT4Video/ShareGPT4Video |
| BurstGPT | ✅ | ✅ | wget https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_without_fails_2.csv |
| Sonnet (deprecated) | ✅ | ✅ | Local file: benchmarks/sonnet.txt |
| Random | ✅ | ✅ | synthetic |
| RandomMultiModal (Image/Video) | 🟡 | 🚧 | synthetic |
| RandomForReranking | ✅ | ✅ | synthetic |
| Prefix Repetition | ✅ | ✅ | synthetic |
| HuggingFace-VisionArena | ✅ | ✅ | lmarena-ai/VisionArena-Chat |
| HuggingFace-MMVU | ✅ | ✅ | yale-nlp/MMVU |
| HuggingFace-InstructCoder | ✅ | ✅ | likaixin/InstructCoder |
| HuggingFace-AIMO | ✅ | ✅ | AI-MO/aimo-validation-aime, AI-MO/NuminaMath-1.5, AI-MO/NuminaMath-CoT |
| HuggingFace-Other | ✅ | ✅ | lmms-lab/LLaVA-OneVision-Data, Aeala/ShareGPT_Vicuna_unfiltered |
| HuggingFace-MTBench | ✅ | ✅ | philschmid/mt-bench |
| HuggingFace-Blazedit | ✅ | ✅ | vdaita/edit_5k_char, vdaita/edit_10k_char |
| Spec Bench | ✅ | ✅ | wget https://raw.githubusercontent.com/hemingkx/Spec-Bench/refs/heads/main/data/spec_bench/question.jsonl |
| Custom | ✅ | ✅ | Local file: data.jsonl |

!!! note

    The datasets mentioned above are all links to datasets on huggingface.
    The dataset's dataset-name should be set to hf.
    For local dataset-path, please set hf-name to its Hugging Face ID like

    

3.2 Run basic benchmark

3.2.1 Online serving

First start serving your model:



Then run the benchmarking script:



If successful, you will see the following output:



3.2.2 Offline Throughput Benchmark



If successful, you will see the following output



3.2.3 Multi-Modal Benchmark







3.2.4 Embedding Benchmark






