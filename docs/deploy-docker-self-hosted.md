# Deploy with Docker Compose

Use this guide to deploy the NVIDIA Nsight Copilot Blueprint with Docker Compose on a single DGX Spark.

This deployment provides the local backend for the Nsight Copilot Visual Studio Code extension. After required images and model artifacts are available on the DGX Spark, the extension can use the self-hosted backend without sending prompts or code to an external service.

First startup on DGX Spark can take tens of minutes — a one-time cost of pulling the container images and downloading the model weights from NGC into `OFFLINE_INFERENCE_DIR`. Restarts reuse the cached weights and skip the download.

On every start, including restarts, the services load the models one at a time, because the single GPU and 128 GB of unified memory (shared between CPU and GPU) cannot hold them all at once; memory is freed between each load. GPT-OSS is the slowest to load — roughly 8 minutes from a warm cache and longer on the first run (engine initialization, compilation, CUDA graph capture, and KV-cache setup); Compose allows it up to 20 minutes before treating the start as failed. The other models are faster.

## Supported hardware

The blueprint is tuned for DGX Spark by default and the setup script auto-detects the host's GPUs to choose a placement layout — no manual configuration is required:

| Host | Layout |
| --- | --- |
| **Single GPU large enough for everything (e.g. DGX Spark, 128 GB unified)** | All models share the one GPU; per-model memory fractions are sized to fit the shared pool. |
| **Multi-GPU with a GPU ≥ 80 GB for the LLM + a second GPU (e.g. 2× H100, dual GH200)** | GPT-OSS gets a dedicated GPU (the largest) with a larger KV cache; the autocomplete (7B), embedding, and reranker models are packed onto a second GPU. Any further GPUs are left idle. |
| **GPUs too small (a single 80 GB GPU, or all GPUs below the floor)** | No automatic layout fits, so the setup script stops with guidance — use suitable hardware or configure tensor-parallel serving manually (it is not set up automatically). |

GPT-OSS runs on a single GPU (TP=1) rather than tensor-parallel across GPUs: it fits on one GPU (the ~63 GB MXFP4 weights plus KV cache; compose caps the context with `NIM_MAX_MODEL_LEN` — set from `GPT_OSS_MAX_MODEL_LEN`, default `32768` — so it runs in ~75 GB on an 80 GB GPU), so the NCCL communication overhead of splitting it outweighs the bandwidth gain. The same reasoning pins the autocomplete model (7B) to a single GPU. On every start the services still load one model at a time so peak memory stays bounded.

## Clone the Repository

```bash
git clone https://github.com/NVIDIA-AI-Blueprints/nsight-copilot.git
cd nsight-copilot
```

## Automated Setup

The setup script handles all prerequisites and starts the services:
```
curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sudo -E sh
sudo -E $(which uv) run deploy/compose/setup_nsc_with_offline_inference.py
```

The script is idempotent — safe to re-run if interrupted.

The script passes the values it resolves — `NGC_API_KEY`, `OFFLINE_INFERENCE_DIR`,
`USER_ID`, and any GPU-placement variables — only to the `docker compose up` that it
runs itself; it does not save them anywhere. The stack it starts is unaffected, but a
later `docker compose` command of your own — after a reboot, or to restart one service
— falls back to the compose defaults. On a multi-GPU host that silently undoes the
detected placement and pins the wrong gpt-oss image. Write the values to
`deploy/compose/.env` so they persist; see [Persist the settings](#persist-the-settings).

If you prefer to set things up manually, follow the steps below.

## Manual Setup

### Prerequisites

1. Install [Docker Engine](https://docs.docker.com/engine/install/ubuntu/) if it is not installed.

2. [Configure](https://docs.docker.com/engine/install/linux-postinstall/) Docker Engine to run without sudo if it is not configured.

3. Install [NVIDIA GPU driver](https://ubuntu.com/server/docs/nvidia-drivers-installation) if it is not installed.

4. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#with-apt-ubuntu-debian) if it is not installed.
   
5. [Configure](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker) NVIDIA Container Toolkit for Docker if it is not configured.

6. Generate an NGC API key if not generated:
   1. Go to [NGC](https://ngc.nvidia.com/setup) and log in.
   2. Select **Setup** > **Generate API Key**.
   3. Copy the key (starts with `nvapi-`).

7. Ensure you have at least **200 GB** of free disk space for Docker images, model weights, caches, and vector database data.

### Start Services

Export the required environment variables:

```bash
export NGC_API_KEY="nvapi-..."
export OFFLINE_INFERENCE_DIR=/some/path/you/like
export USER_ID=$(id -u)
```

- `NGC_API_KEY` is required to download models from NGC and to pull NIM container images.
- `OFFLINE_INFERENCE_DIR` is mounted to the containers and used to store model weights, so they don't need to be re-downloaded on subsequent runs.
- `USER_ID` is passed to the NIM containers so they run as the host user, avoiding permission issues on the mounted cache volumes.

Compose creates missing `OFFLINE_INFERENCE_DIR` bind sources automatically. If startup fails with a permission error while writing model caches, or if Docker-created paths are owned by root, fix ownership and restart:

```bash
sudo mkdir -p "${OFFLINE_INFERENCE_DIR}"
sudo chown -R "$(id -u):$(id -g)" "${OFFLINE_INFERENCE_DIR}"
```

#### Platform-specific settings

The automated setup script auto-detects the host's GPUs and exports the
environment variables below before `docker compose up`. In manual mode you set
them yourself. The compose defaults target **DGX Spark** (a single unified-memory
GPU), so **on DGX Spark you can skip this section**. On other hardware, export the
variables that apply to your host alongside the ones above.

Check your GPUs first:

```bash
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
```

**1. gpt-oss NIM image tag — `GPT_OSS_IMAGE_TAG`**

The gpt-oss NIM version is host-conditional:

| Host | Setting |
|---|---|
| DGX Spark / any unified-memory GPU (GB10 — memory shows `[N/A]`) | Leave unset (default `1.6.1`). NIM 2.0.x can exhaust the 128 GB unified memory pool and hang the host — there is no memory limiter to bound it. |
| Discrete GPUs (H100, GH200, …) | `export GPT_OSS_IMAGE_TAG=2.0.7` — 1.6.1 does not run on discrete GPUs. |

**2. Multi-GPU placement**

This layout applies only when **one** GPU is **Hopper or newer** (compute
capability ≥ 9.0) **and ≥ 80 GB** to hold the LLM, **and a second** GPU is **≥ 24
GB** for the small models. Dedicate that largest GPU to the LLM and put
autocomplete, embedding, and reranking on the second — using each GPU's `index`
column from `nvidia-smi`. Example — GPU `0` is the ≥ 80 GB LLM GPU, GPU `1` holds
the small models:

```bash
export GPT_OSS_GPU=0
export GPT_OSS_KVCACHE_PERCENT=0.9
export AUTOCOMPLETE_GPU=1
export AUTOCOMPLETE_GPU_MEM_UTIL=0.3
export BGE_GPU=1
export BGE_GPU_MEM_UTIL=0.05
export RERANK_GPU=1
```

On a **single-GPU** host (DGX Spark), leave all of these unset — the defaults run
every model on GPU `0` with Spark-tuned memory fractions.

If **no single GPU clears 80 GB** (e.g. several 48 GB cards), this blueprint has no
supported layout: the LLM would need tensor-parallel serving across GPUs, which is
not configured here and must be set up manually. The automated setup script stops
with guidance in that case.

**3. Other optional variables**

| Variable | Default | Purpose |
|---|---|---|
| `NSC_PORT` | `8080` | Host port for the Nsight Copilot server. Change it if `8080` is taken; set `Server Origin` to match. |
| `GPT_OSS_MAX_MODEL_LEN` | `32768` | Context cap for gpt-oss, passed to the NIM as `NIM_MAX_MODEL_LEN`. Raising it increases KV-cache memory. |

**4. Reranker model profile on GH200 — `NIM_MODEL_PROFILE`**

On GH200 (aarch64 + Hopper, cc 9.0) the only cc-9.0 reranker TensorRT plan is
x86_64-built and fails to deserialize, and NIM auto-selects it anyway. Pin the
portable ONNX profile:

```bash
export NIM_MODEL_PROFILE=f7391ddbcb95b2406853526b8e489fedf20083a2420563ca3e65358ff417b10f
```

Leave unset on x86_64 and on other aarch64 hosts (DGX Spark GB10, GB200) — NIM
auto-selects a loadable profile there.

#### Persist the settings

`export` lasts only as long as the shell. Compose also reads a `.env` file next to
`compose.yaml`, so writing the variables to `deploy/compose/.env` applies them to every
later `docker compose` command, including after a reboot:

```bash
cat > deploy/compose/.env <<EOF
NGC_API_KEY=${NGC_API_KEY}
OFFLINE_INFERENCE_DIR=${OFFLINE_INFERENCE_DIR}
USER_ID=$(id -u)
EOF
chmod 600 deploy/compose/.env
```

Add any platform-specific variables from the section above to the same file. Without
them, an unset `NGC_API_KEY` blocks image and model pulls, an unset `USER_ID` leaves the
NIM containers with an empty `user:` value, and unset placement variables put every
model back on GPU `0`.

The file holds your NGC key, so it is listed in `.gitignore` — keep it out of commits.

Authenticate Docker with the NGC container registry:

```bash
echo "${NGC_API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

Start all services:

```bash
docker compose -f deploy/compose/compose.yaml up -d
```

### Verify the Deployment

Track progress with `docker compose -f deploy/compose/compose.yaml ps`. The GPU
services (`gpt-oss-nim`, `autocomplete-nim`, `vllm-bge`, `rerank-nim`), `litellm`,
`rag-server-offline`, and the Milvus services report `healthy` when they are ready. Two
kinds of service never do, by design: the one-shot `model-prep`, `*-nim-cache`,
`cache-flush-*`, and `test-*` steps run to completion and show `Exited (0)`, and
`nsight-copilot-offline` — the primary backend on port `8080` — defines no healthcheck,
so it only ever shows `Up`.

Do not use that service's `/healthcheck` endpoint as a readiness signal. It probes a
database and a Redis instance that the offline profile does not deploy, so it answers
`{"status":"unhealthy"}` on a fully working stack. Confirm the backend with the sample
requests below instead — `nsight-copilot-offline` starts only after `litellm` reports
healthy, which is the last link in the startup chain.

Once the stack is up, verify the services with sample requests.

#### Chat (GPT-OSS)

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-120b medium reasoning + CUDA knowledge",
    "messages": [
      {"role": "user", "content": "Write a CUDA kernel that applies ReLU to a float array."}
    ],
    "max_tokens": 1024
  }'
```

#### Code Completion

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/CUDA-autocomplete",
    "prompt": "<|fim_prefix|>__global__ void relu_kernel(float* out, const float* in, int n) {\n    int i = blockIdx.x * blockDim.x + threadIdx.x;\n    if (i < n) {\n        out[i] = <|fim_suffix|>;\n    }\n}\n<|fim_middle|>",
    "max_tokens": 32,
    "temperature": 0.01,
    "stop": ["<|endoftext|>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"]
  }'
```

#### Reranking

```bash
curl -X POST "http://localhost:8020/v1/ranking" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/llama-nemotron-rerank-1b-v2",
    "query": {
      "text": "What is the GPU memory bandwidth of H100 SXM?"
    },
    "passages": [
      {
        "text": "Accelerated servers with H100 deliver the compute power along with 3 terabytes per second (TB/s) of memory bandwidth per GPU and scalability with NVLink and NVSwitch."
      },
      {
        "text": "A100 provides up to 20X higher performance over the prior generation and can be partitioned into seven GPU instances to dynamically adjust to shifting demands."
      }
    ]
  }'
```

## Use the Blueprint

Connect a client to the local backend to start using the blueprint.

### Connect from IDE

#### Visual Studio Code
After the services are running on DGX Spark, connect from Visual Studio Code:

1. Install [NVIDIA Sync](https://docs.nvidia.com/dgx/dgx-spark/nvidia-sync.html).
2. Connect [NVIDIA Sync](https://docs.nvidia.com/dgx/dgx-spark/nvidia-sync.html) to DGX Spark.
3. Select **VSCode** from NVIDIA Sync Apps. For more information, see NVIDIA's [Visual Studio Code on DGX Spark](https://build.nvidia.com/spark/vscode/overview)
   guide.
4. After Visual Studio Code starts, install the
   [Nsight Copilot Visual Studio Code extension](https://marketplace.visualstudio.com/items?itemName=NVIDIA.nsight-copilot)
   from the Visual Studio Marketplace.
5. Configure the Nsight Copilot extension to use the self-hosted backend: In Visual Studio Code, click on the Extensions tab -> find Nsight Copilot -> click on its gear icon -> click on Settings -> change `Server Origin` value to `http://localhost:8080`
6. Forward DGX Spark port `8080` to local port `8080`: In Visual Studio Code,
   open **Terminal** > **New Terminal**, switch to the **Ports** tab, and forward port `8080`.
   For more options, see the Visual Studio Code documentation for
   [port forwarding](https://code.visualstudio.com/docs/debugtest/port-forwarding) and
   [Remote SSH](https://code.visualstudio.com/docs/remote/ssh).

#### VS Code-compatible forks

Cursor, Windsurf, and VSCodium may work with additional extension-install and port-forwarding adjustments. These clients may not appear as native NVIDIA Sync apps. If the Visual Studio Marketplace is not available in the client, install the [Nsight Copilot extension from Open VSX](https://open-vsx.org/extension/NVIDIA/nsight-copilot), then configure `Server Origin` to `http://localhost:8080` and forward DGX Spark port `8080` to local port `8080`.

### Forward ports over SSH

If you reach the host from a plain terminal (any remote server, not only through
NVIDIA Sync), forward the ports you need with `ssh -L`. The primary backend is
port `8080`:

```bash
ssh -L 8080:localhost:8080 <user>@<host>
```

Keep that session open; while it runs, `http://localhost:8080` on your machine
reaches the server on the host. Point the extension `Server Origin` (or your API
client) at `http://localhost:8080`.

To forward several ports at once — for example to hit an individual NIM directly —
repeat `-L` (see [Service Ports](#service-ports) for the host ports):

```bash
# 8080 = primary backend, 8020 = rerank NIM
ssh -L 8080:localhost:8080 -L 8020:localhost:8020 <user>@<host>
```

Run the tunnel in the background without opening a shell with `-fN`:

```bash
ssh -fN -L 8080:localhost:8080 <user>@<host>
```

`-L <local>:localhost:<remote>` maps a local port to that port on the host, resolved
from the host's own perspective — `localhost` in the second field is the remote
machine, not yours. If a local port is taken, pick another and adjust `Server Origin`
to match — e.g. `-L 8081:localhost:8080` with `Server Origin` set to
`http://localhost:8081`.

Compose publishes these ports on every interface of the host (`0.0.0.0`), not on
loopback only, so the tunnel is a convenience rather than the only route in — see
[Local Data and Privacy](#local-data-and-privacy).

Verify a forwarded service from your machine — for example the rerank NIM on `8020`:

```bash
curl -s http://localhost:8020/v1/health/ready && echo OK
```

### Model Context Protocol (MCP)

The Nsight Copilot server exposes an MCP endpoint at `http://localhost:8080/mcp/cuda-docs/`
(once port `8080` is forwarded from DGX Spark). Include the trailing slash: the server
answers the slashless form with a `307` redirect, and not every MCP client replays the
request body when it follows one. It provides a single tool,
`search_cuda_docs`, that performs semantic search over a curated corpus of NVIDIA CUDA
documentation and code samples — current through early 2026, including libraries that
post-date most model training cutoffs (such as cuTile). An MCP-capable client can call
it to ground answers in current, authoritative CUDA material.

#### Flow

1. An MCP client (your IDE or agent) connects to `http://localhost:8080/mcp/cuda-docs/`.
2. The client calls `search_cuda_docs` with a natural-language query.
3. The server retrieves matching documentation and code samples from the RAG
   backend, reranks them, and returns the context to the client.

## Service Ports

| Service | Host Port | Description |
|---------|-----------|-------------|
| LiteLLM unified proxy | 8000 | OpenAI-compatible gateway for internal services |
| GPT-OSS NIM | 8010 | Direct chat/reasoning NIM endpoint |
| CUDA autocomplete NIM | 8011 | Direct fill-in-the-middle code completion endpoint |
| BGE embedding (vLLM) | — | Not published to the host; reachable from other containers at `http://vllm-bge:8000` |
| Rerank NIM | 8020 | Direct reranking endpoint |
| RAG server | 8030 | Direct CUDA document retrieval backend |
| Nsight Copilot server | 8080 | Primary backend API for IDE clients and API users |
| Milvus | 19530, 9091 | Vector database and health endpoint |
| MinIO | 9000, 9001 | Object storage API and console |

## Local Data and Privacy

In the default offline Compose deployment, prompts and code are served by local containers. The stack uses NGC for image and model downloads, and `OFFLINE_INFERENCE_DIR` for local model caches. Do not override the model endpoints to external services unless your deployment policy allows code or prompt data to leave the machine.

Local inference is not the same as a closed network. Compose publishes every port in [Service Ports](#service-ports) on all of the host's interfaces (`0.0.0.0`), and none of these services authenticate requests. Anyone who can reach the host can send prompts to port `8080`, query the NIMs directly, read the Milvus collections on `19530`, and open the MinIO console on `9001` using its default `minio_admin` / `minio_admin` credentials. On a shared or untrusted network, restrict that surface: bind the published ports to `127.0.0.1` in `deploy/compose/compose.yaml` and reach them over an SSH tunnel, or firewall them to the hosts that need access. If the object store stays reachable, change its credentials with `MINIO_ACCESS_KEY_ID` and `MINIO_SECRET_ACCESS_KEY`.

## Troubleshooting

| Symptom | Likely cause | What to check |
|---------|--------------|---------------|
| `docker pull` or NIM startup returns unauthorized | NGC key lacks required access | Re-run `docker login nvcr.io` and confirm the NGC key can pull `nvcr.io/nim/*` and `nvcr.io/nvidia/blueprint/*` images. |
| Startup is slow or never completes | Services start one at a time (DGX Spark holds one model at a time), each gated on the previous passing its healthcheck plus a cache flush; first run also pulls model weights from NGC. A NIM that fails to initialize halts the whole chain and the main server never starts. | `docker compose ps` to see which service is currently `starting` or stuck `unhealthy` (or a `cache-flush`/`test-*` step that did not complete); `docker compose logs -f <service>` to follow it. First-run init is budgeted (GPT-OSS up to ~20 minutes) — if it is still progressing, wait; a service past its budget has genuinely failed. |
| Disk fills during setup | Docker images plus model caches exceed available space | Meet the Prerequisites free-disk requirement; move `OFFLINE_INFERENCE_DIR` to a larger disk if needed. |
| NIM reports no compatible profile or memory pressure | Host GPU layout not recognized, or GPUs too small to hold a model on one device | The setup script auto-detects GPUs and tunes placement for single-GPU (DGX Spark) and multi-GPU hosts with a GPU ≥ 80 GB for the LLM plus a second GPU for the small models (see [Supported hardware](#supported-hardware)). When the GPUs are too small, the script stops — use suitable hardware or configure tensor-parallel serving manually. Confirm `nvidia-smi` reports the expected GPUs and memory; the setup script prints the chosen GPU placement profile during step 6. |
| GPU containers cannot see GPUs | NVIDIA Container Toolkit missing or unconfigured | Run `docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi`. If it fails, see the [NVIDIA Container Toolkit troubleshooting guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html). |
| No autocomplete suggestions in the IDE | Backend URL, port forwarding, or autocomplete service issue, or autocomplete toggle is off | Confirm the extension `Server Origin` is `http://localhost:8080`; confirm port `8080` is forwarded from DGX Spark to the client machine; check `docker compose ps autocomplete-nim litellm` shows both as `healthy`; verify autocomplete is enabled — click **Nsight Copilot** in the VS Code status bar (bottom-right) and ensure **Enable autocomplete** is on. |
| Autocomplete suggestions are poor, distracting, or contain stray tokens | File language outside the model's primary training scope (CUDA, C++, Python), or autocomplete is not wanted for this session | The autocomplete model is fine-tuned mainly on CUDA, C++, and Python code; completion quality on other languages is limited. To disable inline suggestions, click **Nsight Copilot** in the VS Code status bar (bottom-right) and toggle **Enable autocomplete** off. The same menu re-enables it. |
| MCP tools unavailable, or `search_cuda_docs` returns no context | The MCP endpoint is not reachable, the trailing slash is missing, or the RAG backend has no data | Confirm the MCP client points at `http://localhost:8080/mcp/cuda-docs/` — with the trailing slash — and that port `8080` is forwarded from DGX Spark to the client machine; confirm `docker compose ps rag-server-offline` is `healthy` and check its logs for retrieval errors. |
| Port already in use | Another service is bound to the same port — either the published Compose port on DGX Spark, or the forwarded local port on the client | Identify which side collides. **On DGX Spark:** run `sudo ss -ltnp 'sport = :8080'` (or `sudo lsof -i :8080`) to find the pre-existing process holding the port; stop that process (recommended — DGX Spark resources are limited, so avoid a competing service) or, if it must keep running, change the published port in `deploy/compose/compose.yaml`. **On the client:** in the VS Code **Ports** tab, stop the process using the local port and re-forward, or edit the **Local Address** to a free port (for example `8081`) and update the extension `Server Origin` to match (`http://localhost:8081`). |
