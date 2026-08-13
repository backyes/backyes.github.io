# Paged Attention

!!! warning
    This is a historical document based on the [original paper for vLLM](https://arxiv.org/abs/2309.06180).
    It no longer describes the code used in vLLM today.

Currently, vLLM utilizes its own implementation of a multi-head query
attention kernel (`csrc/attention/attention_kernels.cu`).
This kernel is designed to be compatible with
vLLM's paged KV caches, where the key and value cache are stored in
separate blocks (note that this block concept differs from the GPU
thread block. So in a later document, I will refer to vLLM paged
attention block as "block", while refer to GPU thread block as
"thread block").

To achieve high performance, this kernel relies on a specially
designed memory layout and access method, specifically when threads
read data from global memory to shared memory. The purpose of this
document is to provide a high-level explanation of the kernel
implementation step by step, aiding those who wish to learn about the
vLLM multi-head query attention kernel. After going through this
document, users will likely have a better understanding and feel easier
to follow the actual implementation.

Please note that this document may not cover all details, such as how
to calculate the correct index for the corresponding data or the dot
multiplication implementation. However, after reading this document
and becoming familiar with the high-level logic flow, it should be
easier for you to read the actual code and understand the details.

## Inputs

The kernel function takes a list of arguments for the current thread
to perform its assigned work. The three most important arguments are
the input pointers `q`, `k_cache`, and `v_cache`, which point
to query, key, and value data on global memory that need to be read
and processed. The output pointer `out` points to global memory
where the result should be written. These four pointers actually
refer to multidimensional arrays, but each thread only accesses the
portion of data assigned to it. I have omitted all other runtime
parameters here for simplicity.

```cpp
template<typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS, int PARTITION_SIZE = 0>
__device__ void paged_attention_kernel(
    ... // Other side args.
    const scalar_t* __restrict__ out,       // [num_seqs, num_heads, max_num_partitions, head_size]
    const scalar_t* __restrict__ q,         // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ k_cache,   // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const scalar_t* __restrict__ v_cache,   // [num_blocks, num_kv_heads, head_size, block_size]
    ... // Other side args.
)
```

There is also a list of template arguments above the function
signature that are determined during compilation time. `scalar_t`
represents the data type of the query, key, and value data elements,
such as FP16. `HEAD_SIZE` indicates the number of elements in each
head. `BLOCK_SIZE` refers to the number of tokens in each block.
`NUM_THREADS` denotes the number of threads in each thread block.
`PARTITION_SIZE` represents the number of tensor parallel GPUs (For
simplicity, we assume this is 0 and tensor parallel is disabled).

With these arguments, we need to perform a sequence of preparations.
This includes calculating the current head index, block index, and
other necessary variables. However, for now, we can ignore these
preparations and proceed directly to the actual calculations. It will
be easier to understand them once we grasp the entire flow.

## Concepts

Just before we dive into the calculation flow, I want to describe a
few concepts that are needed for later sections. However, you may
skip this section and return later if you encounter any confusing
terminologies.

- **Sequence**: A sequence represents a client request. For example,
  the data pointed to by `q` has a shape of
  `[num_seqs, num_heads, head_size]`. That represents there are total
  `num_seqs` of query sequence data are pointed by `q`. Since this
  kernel is a single query attention kernel, each sequence only has one
  query token. Hence, the `num_seqs` equals the total number of tokens
  that are processed in the batch.
- **Context**: The context consists of the generated tokens from the
  sequence. For instance, `["What", "is", "your"]` are the context
  tokens, and the input query token is `"name"`. The model might
  generate the token `"?"`.
- **Vec**: The vec is a list of elements that are fetched and
  calculated together. For query and key data, the vec size
  (`VEC_SIZE`) is determined so that each thread group can fetch and
  calculate 16 bytes of data at a time. For value data, the vec size
  (`V_VEC_SIZE`) is determined so that each thread can fetch and
  calculate 16 bytes of data at a time. For example, if the
  `scalar_t` is FP16 (2 bytes) and `THREAD_GROUP_SIZE` is 2, the
  `VEC_SIZE` will be 4, while the `V_VEC_SIZE` will be 8.
- **Thread group**: The thread group is a small group of
  threads(`THREAD_GROUP_SIZE`) that fetches and calculates one
  query token and one key token at a time. Each thread handles only a
  portion of the token data. The total number of elements processed by
  one thread group is referred as `x`. For example, if the thread
  group contains 2 threads and the head size is 8, then thread 0
  handles the query and key elements at index 0, 2, 4, 6, while thread
  1 handles the elements at index 1, 3, 5, 7.
- **Block**: The key and value cache data in vLLM are split into
  blocks. Each block stores data for a fixed number(`BLOCK_SIZE`)
  of tokens at one head. Each block may contain only a portion of the
  whole context tokens. For example, if the block size is 16 and the
  head size is 128, then for one head, one block can store 16 * 128 =
  2048 elements.
- **Warp**: A warp is a group of 32 threads(`WARP_SIZE`) that
  execute simultaneously on a stream multiprocessor (SM). In this
  kernel, each warp processes the calculation between one query token
  and key tokens of one entire block at a time (it may process multiple
  blocks in multiple iterations). For example, if there are 4 warps and
  6 blocks for one context, the assignment would be like warp 0 handles
  the 0th, 4th blocks, warp 1 handles the 1st, 5th blocks, warp 2
  handles the 2nd block and warp 3 handles the 3rd block.
- **Thread block**: A thread block is a group of
  threads(`NUM_THREADS`) that can access the same shared memory.
  Each thread block contains multiple warps(`NUM_WARPS`), and in
  this kernel, each thread block processes the calculation between one
  query token and key tokens of a whole context.
- **Grid**: A grid is a collection of thread blocks and defines the
  shape of the collection. In this kernel, the shape is
  `(num_heads, num_seqs, max_num_partitions)`. Therefore, each thread
  block only handles the calculation for one head, one sequence, and
  one partition.

## Query

This section will introduce how query data is stored in memory and
fetched by each thread. As mentioned above, each thread group fetches
one query token data, while each thread itself only handles a part of
one query token data. Within each warp, every thread group will fetch
the same query token data, but will multiply it with different key
token data.

```cpp
const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
```

![query](../assets/design/paged_attention/query.png)

Each thread defines its own `q_ptr` which points to the assigned
query token data on global memory. For example, if `VEC_SIZE` is 4
and `HEAD_SIZE` is 128, the `q_ptr` points to data that contains
total of 128 elements divided into 128 / 4 = 32 vecs.

![q_vecs](../assets/design/paged_attention/q_vecs.png)

```cpp
__shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
```

Next, we need to read the global memory data pointed to by `q_ptr`
into shared memory as `q_vecs`. It is important to note that each
vecs is assigned to a different row. For example, if the
`THREAD_GROUP_SIZE` is 2, thread 0 will handle the 0th row vecs,
while thread 1 handles the 1st row vecs. By reading the query data in
this way, neighboring threads like thread 0 and thread 1 can read
neighbor memory, achieving the memory coalescing to improve
performance.

## Key

Similar to the "Query" section, this section introduces memory layout
and assignment for keys. While each thread group only handle one
query token one kernel run, it may handle multiple key tokens across
multiple iterations. Meanwhile, 