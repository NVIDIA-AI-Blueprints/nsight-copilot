<h1>NVIDIA AI Blueprint: Nsight Copilot</h1>

GPU development demands deep expertise across CUDA, parallel computing, and performance optimization. Developers frequently context-switch between documentation, code examples, and best practices scattered across multiple sources. Traditional code assistants lack the specialized knowledge required for high-performance GPU programming.

This blueprint deploys **Nsight Copilot** on DGX Spark — a self-hosted backend for the Nsight Copilot Visual Studio Code extension that delivers expert-level, contextually aware answers to complex CUDA challenges, generates optimized CUDA snippets and kernels from natural language descriptions, and grounds every response in authoritative CUDA documentation via retrieval-augmented generation. Developers can run the backend locally on DGX Spark and connect the IDE extension without sending prompts or code to an external service. Benchmarked using the **[ComputeEval](https://github.com/NVIDIA/compute-eval)** framework for assessing CUDA-related task proficiency.

> **Third-Party Software Notice**
> This project will download and install additional third-party open source software projects.
> Please review the license terms of these open source projects before use.


## Architecture Diagram

![Architecture Diagram](https://assets.ngc.nvidia.com/products/api-catalog/nsight-copilot/diagram.jpg)


## Key Features

- **Expert CUDA-Aware Chat** — Multi-turn conversational AI with OpenAI-compatible streaming that delivers expert-level answers to complex CUDA challenges — from architectural best practices to deep-dive conceptual explanations.
- **CUDA Code Generation and Autocompletion** — Generate complex, optimized CUDA snippets and kernels from natural language descriptions. Real-time inline code completions powered by the `nvidia/CUDA-autocomplete` model provide low-latency suggestions with minimal time-to-first-token.
- **Interactive Code Transformation** — Directly modify and optimize CUDA code in the editor — refactoring for efficiency, converting PyTorch operations into optimized CUDA kernels, and ensuring compatibility with NVIDIA technologies.
- **CUDA Knowledge Retrieval (RAG)** — Retrieval-augmented generation powered by Bodhi Tree RAG surfaces relevant documentation, code examples, and best practices from an authoritative CUDA knowledge corpus — including CUDA Toolkit documentation, programming guides, and optimization references.
- **Supported IDE Clients** — Visual Studio Code and compatible forks; see [Use the Blueprint](#use-the-blueprint) for the full list and connection points.


## Software Components

### NVIDIA Models

- [gpt-oss-120b NIM](https://build.nvidia.com/nvidia/gpt-oss-120b) — LLM for chat and RAG-augmented code generation
- [nvidia/CUDA-autocomplete](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/blueprint/models/cuda-autocomplete) — Specialized model for real-time CUDA code completion
- [llama-nemotron-rerank-1b-v2 NIM](https://build.nvidia.com/nvidia/llama-nemotron-rerank-1b-v2) — Reranking model for retrieval relevance

### Models

- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) — Embedding model for CUDA knowledge corpus

### Infrastructure

- [vLLM](https://vllm.ai/) — High-performance model serving
- [LiteLLM](https://www.litellm.ai/) — Unified model routing proxy
- [FastAPI](https://fastapi.tiangolo.com/) — Async web framework with SSE streaming


## Minimum System Requirements

### Hardware Requirements
- An NVIDIA GPU setup that can host the models — the setup script auto-detects the GPUs and picks the layout, and aborts early with guidance if the host is too small. One of:
  - **DGX Spark** (single GB10, 128 GB) — the reference setup; or
  - a **multi-GPU** host with one GPU **≥ 80 GB** for the LLM plus a second GPU for the smaller models (autocomplete, embedding, reranker) — e.g. 2× H100, dual GH200, H200; or
  - a **single GPU** large enough for the whole stack (~100 GB+, e.g. GH200).
  - The LLM GPU must be **Hopper or newer** (compute capability ≥ 9.0) — gpt-oss is MXFP4, so Ampere (e.g. A100) isn't supported regardless of memory.
  - A single 80 GB GPU on its own is **not** enough — it fits the LLM but not the LLM and the other models together. Splitting the LLM across several smaller GPUs (tensor parallelism) is not configured automatically.
- At least 200 GB of free disk space for Docker images, model weights, caches, and vector database data

### OS Requirements
- Ubuntu 22.04+

### Software Requirements
- NVIDIA GPU driver
- Docker with Compose v2
- NVIDIA Container Toolkit


## Get Started

The recommended way to get started is to deploy the blueprint with Docker Compose on a DGX Spark.
For details, refer to [Deploy with Docker Compose](docs/deploy-docker-self-hosted.md).


## Use the Blueprint

After deployment, connect a client to the local backend running on the DGX Spark. The primary client is the Nsight Copilot Visual Studio Code extension; VS Code-compatible forks may also work. For the supported clients and step-by-step setup, see [Connect from IDE](docs/deploy-docker-self-hosted.md#connect-from-ide).

The default offline Compose deployment serves prompts and code locally through containers on the DGX Spark. NGC access is used for image and model downloads; do not override local model endpoints to external services unless your deployment policy allows code or prompt data to leave the machine.


## Contributing

This project is not currently open to external code contributions. We welcome bug reports, feature requests, and feedback — please file an issue.


## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility, and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their supporting model team to ensure the models meet requirements for the relevant industry and use case and address unforeseen product misuse. For more detailed information on ethical considerations for the models, please see the Model Card++ Explainability, Bias, Safety & Security, and Privacy Subcards. Please report security vulnerabilities or NVIDIA AI concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).


## License

This NVIDIA AI Blueprint is licensed under the [Apache License, Version 2.0](./LICENSE). This project will download and install additional third-party open source software projects and containers. Review the license terms of these open source projects before use.

Use of the models in this blueprint is governed by the [NVIDIA AI Foundation Models Community License](https://docs.nvidia.com/ai-foundation-models-community-license.pdf).


## Terms of Use

GOVERNING TERMS: This blueprint uses the following components, which are governed by the terms listed below:

### Nsight Copilot
Use of Nsight Copilot is governed by [NVIDIA Technology Access Terms of Use](https://docs.nvidia.com/cuda/eula/index.html#cuda-toolkit-supplement-to-software-license-agreement-for-nvidia-software-development-kits) and the CUDA content is governed by the License Agreement for NVIDIA Software Development Kits and CUDA Toolkit Supplement to Software License Agreement for NVIDIA Software Development Kits.

### gpt-oss-120b & llama-nemotron-rerank-1b-v2 NIM Containers
Use of gpt-oss-120b & llama-nemotron-rerank-1b-v2 NIM containers is governed by the [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/) and [Product-Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/).

### gpt-oss-120b, bge-m3, llama-nemotron-rerank-1b-v2 & cuda-autocomplete models
Use of gpt-oss-120b, bge-m3, llama-nemotron-rerank-1b-v2 & cuda-autocomplete models is governed by the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).

### ADDITIONAL INFORMATION
gpt-oss-120b model is licensed under [Apache License, Version 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md).

llama-nemotron-rerank-1b-v2 is licensed under [Llama 3.2 Community Model License Agreement](https://www.llama.com/llama3_2/license). Built with Llama. 

CUDA Autocomplete model is based on Qwen2.5-Coder-7B model, which is licensed under [Apache License, Version 2.0](https://huggingface.co/Qwen/Qwen2.5-Coder-7B/blob/main/LICENSE).
