# Nsight Copilot 26.3.1 (3 September 2026)

Various improvements

## New Features

-

## Improvements

- Prompt truncation improvements make chat completions run faster.
- Log response body for failed RAG API calls.
- Improved input validation for RAG API calls.
- Upgraded python dependencies.
- Improved RAG error reporting.
- Updated RAG corpus.
- Improved input validation for MCP calls.
- Use compose healthcheck service for Nsight Copilot server.

## Bug Fixes
- Healthcheck endpoint no longer polls components that aren't part of the deployment.

# Nsight Copilot 26.2.1 (19 July 2026)

Multi-GPU support

## New Features

- Support for multi-GPU and heterogeneous GPU configurations, including H100, H200, GH200, and DGX Spark.

## Improvements

- MCP returns more accurate cuTile results.
- Upgraded python dependencies.

## Bug Fixes

- Don't crash nor allow empty str input for RAG.
- Remove unused python dependencies.
- Adjust autocomplete model default config and use it if config is not provided by the client.
- Remove meaningless warnings during RAG boot.

# Nsight Copilot 26.1.1 (04 Jun 2026)

Initial public release of the NVIDIA AI Blueprint: Nsight Copilot — a self-hosted,
offline backend for the Nsight Copilot Visual Studio Code extension, deployed with
Docker Compose on NVIDIA DGX Spark.

## New Features

- Self-hosted backend for the Nsight Copilot VS Code extension, exposing four
  capabilities to connected IDE clients:
    - Expert CUDA-aware chat — multi-turn, OpenAI-compatible streaming for architectural
      guidance and deep-dive explanations.
    - Code generation and autocompletion — natural-language-to-CUDA snippets and kernels,
      plus low-latency inline completions.
    - Interactive code transformation — in-editor refactoring, optimization, and
      PyTorch-to-CUDA kernel conversion.
    - CUDA knowledge retrieval (RAG) — answers grounded in an authoritative CUDA
      documentation corpus via Bodhi Tree RAG.
- Docker Compose deployment (`deploy/compose/compose.yaml`) that brings up the full stack
  on a single DGX Spark, via either a one-command automated setup
  (`setup_nsc_with_offline_inference.py`) or a documented manual path. Inference runs
  entirely on the DGX Spark — prompts and code never leave the machine; NGC access is
  used only to download images and model weights.
- Models and serving stack — gpt-oss-120b NIM, llama-nemotron-rerank-1b-v2 NIM,
  `BAAI/bge-m3` embeddings, and the `nvidia/CUDA-autocomplete` model, served through vLLM
  with a LiteLLM routing proxy and a FastAPI backend.
- IDE integration — connection instructions for the Nsight Copilot VS Code extension
  (and compatible forks), including Model Context Protocol (MCP) support.
- Documentation — [Deploy with Docker Compose](docs/deploy-docker-self-hosted.md)
  covering prerequisites, setup, deployment verification, IDE connection, service ports,
  local-data/privacy notes, and troubleshooting; model cards
  (`deploy/compose/MODEL-CARDS.md`); and third-party license notices (`THIRD-PARTY.txt`).
- Repository metadata — issue templates (bug report, feature request, new and correction
  documentation requests), Code of Conduct, and security policy.

## Improvements

- None; initial release.

## Bug Fixes

- None; initial release.
