<h1>NVIDIA AI Blueprint: Nsight Copilot</h1>

GPU development demands deep expertise across CUDA, parallel computing, and performance optimization. Developers frequently context-switch between documentation, code examples, and best practices scattered across multiple sources. Traditional code assistants lack the specialized knowledge required for high-performance GPU programming.

This blueprint deploys **Nsight Copilot** — a self-hosted Model Context Protocol (MCP) server that grounds AI coding assistants and agents in authoritative CUDA documentation. Any MCP-capable client can call its `search_cuda_docs` tool to retrieve relevant documentation and code samples from a curated CUDA knowledge corpus. Developers can run the server locally without sending prompts or code to an external service.

> **Third-Party Software Notice**
> This project will download and install additional third-party open source software projects.
> Please review the license terms of these open source projects before use.


## Architecture Diagram

![Architecture Diagram](https://assets.ngc.nvidia.com/products/api-catalog/nsight-copilot/diagram.jpg)


## Key Features

- **CUDA Documentation Search over MCP** — The `search_cuda_docs` tool at `/mcp/cuda-docs` performs semantic search over a curated corpus of NVIDIA CUDA documentation and code samples — current through mid 2026, including libraries that post-date most model training cutoffs (such as cuTile).
- **CUDA Knowledge Retrieval (RAG)** — Retrieval powered by Bodhi Tree RAG surfaces relevant documentation, code examples, and best practices from an authoritative CUDA knowledge corpus — including CUDA Toolkit documentation, programming guides, and optimization references.
- **Supported Clients** — Any MCP-capable client (IDE assistant or agent); see [Use the Blueprint](#use-the-blueprint) for the connection point.


## Software Components

### NVIDIA Models

- [llama-nemotron-rerank-1b-v2 NIM](https://build.nvidia.com/nvidia/llama-nemotron-rerank-1b-v2) — Reranking model for retrieval relevance

### Models

- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) — Embedding model for CUDA knowledge corpus

### Infrastructure

- [vLLM](https://vllm.ai/) — High-performance model serving
- [LiteLLM](https://www.litellm.ai/) — Unified model routing proxy
- [FastAPI](https://fastapi.tiangolo.com/) — Async web framework with SSE streaming


## Minimum System Requirements

### Hardware Requirements
- An NVIDIA GPU that can host the models — all models run on GPU `0`; the setup script auto-detects the GPUs and aborts early with guidance if GPU `0` is too small. One of:
  - **DGX Spark** (single GB10, 128 GB) — the reference setup; or
  - a discrete GPU `0` with **≥ 24 GB** of memory (e.g. H100, GH200).
- At least 100 GB of free disk space for Docker images, model weights, caches, and vector database data

### OS Requirements
- Ubuntu 22.04+

### Software Requirements
- NVIDIA GPU driver
- Docker with Compose v2
- NVIDIA Container Toolkit


## Get Started

The recommended way to get started is to deploy the blueprint with Docker Compose.

For details, refer to [Deploy with Docker Compose](docs/deploy-docker-self-hosted.md).


## Use the Blueprint

After deployment, connect an MCP client to the local server.

For details, see [Model Context Protocol (MCP)](docs/deploy-docker-self-hosted.md#model-context-protocol-mcp).

The default offline Compose deployment serves locally through containers. 

NGC access is used for image and model downloads; do not override local model endpoints to external services unless your deployment policy allows code or prompt data to leave the machine.


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

### llama-nemotron-rerank-1b-v2 NIM Container
Use of the llama-nemotron-rerank-1b-v2 NIM container is governed by the [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/) and [Product-Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/).

### bge-m3 & llama-nemotron-rerank-1b-v2 models
Use of bge-m3 & llama-nemotron-rerank-1b-v2 models is governed by the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).

### ADDITIONAL INFORMATION
llama-nemotron-rerank-1b-v2 is licensed under [Llama 3.2 Community Model License Agreement](https://www.llama.com/llama3_2/license). Built with Llama. 
