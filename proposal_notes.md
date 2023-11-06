# Machine Learning For Optimal Kernel Generation

# Thesis Goals Listing

## Major Goals

- Completely explore what a deep learning framework is, using a **minimalistic** but production ready approach [Tinygrad](https://github.com/tinygrad/tinygrad/tree/0c9b4ab885103c20867e911854ab87d5ea4c718d). Due to its extreme simplicity, it aims to be the easiest framework to add new accelerators to, with support for both inference and training. If XLA is CISC, tinygrad is RISC. It also offers a fully **Pythonic ecosystem**, which for compiler design and AI is extremely desirable.
- Explore neural networks custom kernel generation for the leading AI industrial hardware accelerators (**GPUs** & **TPUs**)
- Research latest open-source neural networks frameworks releases, such as **PyTorch** 2.1, which strong focus on its new AI compiler.
- Develop profound intuition regarding **array languages and AI compilers** (_e.g._ **Linearization**)
- Compare and research novel open-source GPU programming languages for neural networks. We primarily focus on **Triton** for GPUs and **Pallas** for TPUs.
- Narrow focus on Transformer kernels optimizations, like **Flash-Attention**. Writing custom CUDA kernels in either C++ or via Triton.
- Profiling, debugging, and optimizing multi-host GPU utilization.

## Minor Goals

- Accelerating **Generative AI**.
- AI models **Software engineering** optimizations at scale.
- Gather better understanding of the available open-source **AI Software infrastructure** at scale.
- Configuring and troubleshooting hardware and operating-systems for maximum performance.
- Digging into third-party source code for debugging and customization.

# Context

The field of natural language processing (NLP) has been transformed by massive pre-trained language models. They are becoming a good candidate to improve the performance on a variety of multimodal domain tasks. They have shown an impressive ability to generate fluent text and showcase novel emergent capabilities not present in small scale language models, like **in-context learning** and "reasoning". Nevertheless, these models are hard to understand and give rise to new scalability challenges. 

This project proposal focuses on the technical Software stack underlying the training procedures and optimization techniques involved in the construction of this large transformers-based languages model. There has been an exponential explosion of software tooling related to standardized and formally tackle the issues that emerge from this vast complexity. Therefore this proposal aims at reviewing, summarizing and showing proof of concept models using these cutting-edge Software stack for this technological challenge.

# Problem Statement


For a wide variety of ML applications, AI compiler code generation does a good job. Nevertheless, for achieving upper limit optimization it is inevitable to hit the compiler’s limitations. In these scenarios, we need to provide a mechanism for going lower into the program generation. Therefore it is mandatory to write **hand-tuned** kernels (CUDA, **Triton**) that outperform the compiler at that given point in time. Furthermore, advances in ML systems research take some time to be incorporated into the compiler, thus with custom kernels we can run ahead. Over time some of those handcrafted custom kernels might end up incorporated into the compiler optimizations. Nevertheless, hand-coded optimised kernels can lead to an exhaustive task, whereas correctly applying search and learning strategies would always outperform any manual optimization. Therefore we plan to leverage novel and classical search algorithms, ranging from dynamic programming up to MuZero for kernel code generation.

# AI Compilers
Avoid hand-optimised kernels backends such as [cuDNN](https://github.com/NVIDIA/cudnn-frontend), [cutlass](https://github.com/NVIDIA/cutlass). Instead be capable of applying optimization methods during kernel code generation stage (e.g. *loop unrolling*)

- Represent neural networks computational graph as a pipeline of operations ([CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/)). Therefore, pipeline transformations naturally lead to model optimization.
- **Triton & CUDA:**  Running large neural networks at scale while maximizing compute efficiency is paramount if we want to make optimal use of our compute resources. Hence, we regularly write bespoke kernels either in Triton or in raw C++ CUDA.
- **Tinygrad** Hand-optimised Deep Learning Model Example: **[Resnet50](https://github.com/tinygrad/tinygrad/blob/ddbc6eecaf3f019d19b9aa1627e14898b34dfb07/examples/handcode_resnet50_opt.py#L13)**

- Quantify custom AI industrial hardware accelerators engines speedups, like NVIDIA **Tensor Cores**, which are exclusively present in **Ampere & Hooper** architecture (See: Developing CUDA Kernels to push _Tensor Cores_  to the absolute limit on NVIDIA **A100** Andrew Kerr, May 21, 2020)

- New tracing approaches like PyTorch [**Dynamo**](https://github.com/pytorch/torchdynamo)

- Build the proper Software infrastructure for better code generation search in the given AI language (_e.g._ [beam search](https://github.com/tinygrad/tinygrad/blob/680cbfdba4d3a899c7eadb4acd62c1c6f53d1002/tinygrad/features/search.py#L99))

- Get the whole **actions space** (dynamic programming and RL terminology), whereas each actions represent a possible **tensor operation**. We identify these actions after have generated a synthetic test dataset with a significant number of possible kernels, thus we try to representative sample the **kernel space**. An example action looks like:
```python
from tinygrad.codegen.kernel import Opt, OptOps

actions = flatten([[Opt(op=OptOps.UPCAST, axis=axis, amt=amt) for amt in [0,2,3,4,7]] for axis in range(6)])
```

The dataset generation script would be similar to the following:
````bash
#!/bin/bash
export LOGOPS=/tmp/ops
rm $LOGOPS

# generate many kernels
PYTHONPATH="." OPT=2 GPU=1 python3 test/external/external_test_opt.py
PYTHONPATH="." OPT=3 GPU=1 python3 test/external/external_test_opt.py
GPU=1 IMAGE=1 python3 test/test_ops.py
FORWARD_ONLY=1 GPU=1 IMAGE=2 python test/test_ops.py
STEPS=3 python3 examples/hlb_cifar10.py
WINO=1 STEPS=3 python3 examples/hlb_cifar10.py
python3 examples/stable_diffusion.py --noshow
python3 examples/llama.py --prompt "hello" --count 5
python3 examples/gpt2.py --count 5
python3 examples/mlperf/model_spec.py
python3 examples/yolov8.py ./test/models/efficientnet/Chicken.jpg
openpilot/go.sh
BIG=1 MPS=1 pytest test/

# sort and uniq
sort -u /tmp/ops > /tmp/sops
ls -lh /tmp/ops /tmp/sops
````

- Leverage **learning & search** for kernel code generation.
- GPU Programming languages for neural networks with small set of operations, thus implying small action space. In this category we primarily study [**Triton**](https://openai.com/research/triton) for CUDA & HIP, and [**Pallas**](https://jax.readthedocs.io/en/latest/pallas/index.html) for TPUs.
- Kernel fusion -> https://github.com/microsoft/DeepSpeed/blob/master/csrc/transformer/inference/csrc/gelu.cu#L656
- Checking kernel **correcteness** (verification) via fuzzers, which decompose source code into templates and fragments, making possible to find compiler bugs (See [comby-decomposer](https://github.com/comby-tools/comby-decomposer)). [Tinygrad compiler testing fuzzer example](https://github.com/tinygrad/tinygrad/blob/0c9b4ab885103c20867e911854ab87d5ea4c718d/test/external/fuzz_shapetracker.py)
- The kernel layout is as follows:

  1. **Globals**
  2. **Locals**
  3. **Reduced loops**
  4. **Unrolled loops**
  5. **Output space**

  

## Kernel Runtime Prediction

**Main Reference:** [Google - Fast or Slow? Predict AI Model Runtime](https://www.kaggle.com/competitions/predict-ai-model-runtime)
An AI model can be represented as a **graph**, where a **node** is a **tensor operation** (e.g. matrix multiplication, convolution, etc), and an **edge** represents a **tensor**. A compilation configuration controls how the compiler **transforms** the graph for a specific optimization pass (See: [*Tensor Programs*, Greg Yang](https://thegregyang.com/#tensorprograms)). We conceive two main types of configurations/optimizations:

- **A layout configuration** control how tensors in the **graph** are laid out in the physical memory, by specifying the dimension order of each input and output of an operation node.
- **A tile configuration** controls the tile size of each fused **subgraph**. 

# Case Study: Tinygrad

Framework source code complexity is highly dense:
```shell
tinygrad %./sz.py 
Name                              Lines    Tokens/Line
------------------------------  -------  -------------
tinygrad/tensor.py                  575           21.9
tinygrad/codegen/kernel.py          425           17.8
tinygrad/codegen/linearizer.py      327           18.9
tinygrad/shape/symbolic.py          302           14.9
tinygrad/lazy.py                    243           18.5
tinygrad/ops.py                     239           14.7
tinygrad/renderer/cstyle.py         189           10.6
tinygrad/helpers.py                 174           14.1
tinygrad/shape/shapetracker.py      169           15.2
tinygrad/mlops.py                   149           14.4
tinygrad/features/image.py          140           18.3
tinygrad/features/search.py         125           14.5
tinygrad/renderer/llvmir.py         123           19.7
tinygrad/renderer/triton.py         111           13.1
tinygrad/shape/view.py              102           20.4
tinygrad/nn/state.py                100           16.2
tinygrad/graph.py                    98           13.5
tinygrad/runtime/ops_gpu.py          95           16.0
tinygrad/nn/__init__.py              92           19.1
tinygrad/runtime/lib.py              89           15.3
tinygrad/runtime/ops_cuda.py         85           15.2
tinygrad/runtime/ops_hip.py          75           12.7
tinygrad/runtime/ops_metal.py        71           11.7
tinygrad/jit.py                      67           15.3
tinygrad/realize.py                  55           14.9
tinygrad/nn/optim.py                 54           17.8
tinygrad/runtime/ops_llvm.py         54            9.4
tinygrad/renderer/wgsl.py            52           12.3
tinygrad/runtime/ops_cpu.py          46           25.3
tinygrad/runtime/ops_webgpu.py       40           15.5
tinygrad/runtime/ops_disk.py         36           18.0
tinygrad/runtime/ops_torch.py        35           27.0
tinygrad/runtime/ops_shm.py          29           12.6
tinygrad/runtime/ops_clang.py        28           12.1
tinygrad/renderer/opencl.py          20            7.5
tinygrad/renderer/cuda.py            14            6.2
tinygrad/renderer/metal.py           14            5.9

tinygrad                       :   1600
tinygrad/codegen               :    752
tinygrad/features              :    265
tinygrad/nn                    :    246
tinygrad/renderer              :    523
tinygrad/runtime               :    683
tinygrad/shape                 :    573

total line count: 4642
```
