# Deploy with Docker Compose

Use this guide to deploy the NVIDIA Nsight Copilot Blueprint with Docker Compose.

This deployment provides a local Model Context Protocol (MCP) server for CUDA documentation search. After required images and model artifacts are available on the server, MCP clients can use the self-hosted server without sending prompts or code to an external service.

First startup can take tens of minutes — a one-time cost of pulling the container images and downloading the model weights from NGC into `OFFLINE_INFERENCE_DIR`. Restarts reuse the cached weights and skip the download.

On every start, including restarts, the services load the models one at a time to avoid memory issues & memory is freed between each load.

## Supported hardware

The blueprint is tuned for DGX Spark by default and the setup script auto-detects the host's GPUs — no manual configuration is required. All models (embedding and reranker) run on GPU `0`; any further GPUs are left idle.

| Host | Layout |
| --- | --- |
| **DGX Spark (128 GB unified)** | Compose defaults; per-model memory fractions are sized to fit the shared pool. |
| **Discrete GPU 0 ≥ 24 GB (e.g. H100, GH200)** | The setup script raises the embedding model memory fraction (`BGE_GPU_MEM_UTIL`). |
| **GPU 0 below 24 GB** | The setup script stops with guidance — use suitable hardware. |

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

7. Ensure you have at least **100 GB** of free disk space for Docker images, model weights, caches, and vector database data.

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

**1. Embedding model memory on discrete GPUs — `BGE_GPU_MEM_UTIL`**

The default (`0.01`) is sized for the DGX Spark 128 GB unified pool. On a discrete
GPU (memory shows a number, not `[N/A]`), give the embedding model about 2 GiB:
set the fraction to 2048 divided by GPU 0 `memory.total` in MiB, rounded up to two
decimals. Example for an 80 GB GPU (81559 MiB):

```bash
export BGE_GPU_MEM_UTIL=0.03
```

**2. Reranker model profile on GH200 — `NIM_MODEL_PROFILE`**

On GH200 (aarch64 + Hopper, cc 9.0) the only cc-9.0 reranker TensorRT plan is
x86_64-built and fails to deserialize, and NIM auto-selects it anyway. Pin the
portable ONNX profile:

```bash
export NIM_MODEL_PROFILE=f7391ddbcb95b2406853526b8e489fedf20083a2420563ca3e65358ff417b10f
```

Leave unset on x86_64 and on other aarch64 hosts (DGX Spark GB10, GB200) — NIM
auto-selects a loadable profile there.

Authenticate Docker with the NGC container registry:

```bash
echo "${NGC_API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

Start all services:

```bash
docker compose -f deploy/compose/compose.yaml up -d
```

### Verify the Deployment

After all services report healthy, you can verify them with sample requests.

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

Connect an MCP client to the local backend to start using the blueprint.

### Connect from IDE

#### Visual Studio Code
After the services are running on DGX Spark, connect from Visual Studio Code:

1. Install [NVIDIA Sync](https://docs.nvidia.com/dgx/dgx-spark/nvidia-sync.html).
2. Connect [NVIDIA Sync](https://docs.nvidia.com/dgx/dgx-spark/nvidia-sync.html) to DGX Spark.
3. Select **VSCode** from NVIDIA Sync Apps. For more information, see NVIDIA's [Visual Studio Code on DGX Spark](https://build.nvidia.com/spark/vscode/overview)
   guide.
4. Add the MCP server: open the Command Palette and run **MCP: Add Server**. Select **HTTP**,
   enter `http://localhost:8080/mcp/cuda-docs`, enter a server ID (for example
   `nsight-copilot`), and select **Global**. For more information, see
   [Add and manage MCP servers in VS Code](https://code.visualstudio.com/docs/copilot/customization/mcp-servers).
5. Forward DGX Spark port `8080` to local port `8080`: In Visual Studio Code,
   open **Terminal** > **New Terminal**, switch to the **Ports** tab, and forward port `8080`.
   For more options, see the Visual Studio Code documentation for
   [port forwarding](https://code.visualstudio.com/docs/debugtest/port-forwarding) and
   [Remote SSH](https://code.visualstudio.com/docs/remote/ssh).
6. Open Chat in agent mode and ask a CUDA question. Approve the `search_cuda_docs`
   tool call when Visual Studio Code asks.

#### Other IDEs

Other MCP-capable IDEs and agents can also connect to
`http://localhost:8080/mcp/cuda-docs`. They are not listed here. See the documentation
of your client for how to add a remote HTTP MCP server.

### Forward ports over SSH

If you reach the host from a plain terminal (any remote server, not only through
NVIDIA Sync), forward the ports you need with `ssh -L`. The primary backend is
port `8080`:

```bash
ssh -L 8080:localhost:8080 <user>@<host>
```

Keep that session open; while it runs, `http://localhost:8080` on your machine
reaches the server on the host. Point your MCP client at
`http://localhost:8080/mcp/cuda-docs`.

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

`-L <local>:localhost:<remote>` maps a local port to the same port on the host's
loopback, where Compose publishes the service. If a local port is taken, pick
another and adjust the MCP client URL to match — e.g. `-L 8081:localhost:8080` with
the client pointed at `http://localhost:8081/mcp/cuda-docs`.

Verify a forwarded service from your machine — for example the rerank NIM on `8020`:

```bash
curl -s http://localhost:8020/v1/health/ready && echo OK
```

### Model Context Protocol (MCP)

The MCP endpoint provides one tool, `search_cuda_docs`. It does semantic search over a
curated corpus of NVIDIA CUDA documentation and code samples — current through early 2026,
including libraries that post-date most model training cutoffs (such as cuTile). The server
retrieves matching documents from the RAG backend, reranks them, and returns them to the client.

## Service Ports

| Service | Host Port | Description |
|---------|-----------|-------------|
| LiteLLM unified proxy | 8000 | OpenAI-compatible gateway for internal services |
| Rerank NIM | 8020 | Direct reranking endpoint |
| RAG server | 8030 | Direct CUDA document retrieval backend |
| Nsight Copilot server | 8080 | MCP endpoint (`/mcp/cuda-docs`) |
| Milvus | 19530, 9091 | Vector database and health endpoint |
| SILO | 9000, 9001 | Object storage API and console |

## Local Data and Privacy

In the default offline Compose deployment, prompts and code are served by local containers. The stack uses NGC for image and model downloads, and `OFFLINE_INFERENCE_DIR` for local model caches. Do not override the model endpoints to external services unless your deployment policy allows code or prompt data to leave the machine.

## Troubleshooting

| Symptom | Likely cause | What to check |
|---------|--------------|---------------|
| `docker pull` or NIM startup returns unauthorized | NGC key lacks required access | Re-run `docker login nvcr.io` and confirm the NGC key can pull `nvcr.io/nim/*` and `nvcr.io/nvidia/blueprint/*` images. |
| Startup is slow or never completes | Services start one at a time (DGX Spark holds one model at a time), each gated on the previous passing its healthcheck plus a cache flush; first run also pulls model weights from NGC. A NIM that fails to initialize halts the whole chain and the main server never starts. | `docker compose ps` to see which service is currently `starting` or stuck `unhealthy` (or a `cache-flush`/`test-*` step that did not complete); `docker compose logs -f <service>` to follow it. First-run init is budgeted — if it is still progressing, wait; a service past its budget has genuinely failed. |
| Disk fills during setup | Docker images plus model caches exceed available space | Meet the Prerequisites free-disk requirement; move `OFFLINE_INFERENCE_DIR` to a larger disk if needed. |
| NIM reports no compatible profile or memory pressure | GPU 0 too small, or `BGE_GPU_MEM_UTIL` not raised on a discrete GPU | All models run on GPU 0 (see [Supported hardware](#supported-hardware)). When GPU 0 is too small, the script stops — use suitable hardware. Confirm `nvidia-smi` reports the expected GPUs and memory; the setup script prints the chosen GPU profile during step 6. |
| GPU containers cannot see GPUs | NVIDIA Container Toolkit missing or unconfigured | Run `docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi`. If it fails, see the [NVIDIA Container Toolkit troubleshooting guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html). |
| MCP tools unavailable, or `search_cuda_docs` returns no context | The MCP endpoint is not reachable, or the RAG backend has no data | Confirm the MCP client uses the endpoint URL from [Use the Blueprint](#use-the-blueprint). If the client runs on a different machine than the host, confirm the SSH tunnel is open: `curl -s http://localhost:8080/healthcheck` on the client machine must succeed; confirm `docker compose ps rag-server-offline` is `healthy` and check its logs for retrieval errors. |
| Port already in use | Another service is bound to the same port — either the published Compose port on DGX Spark, or the forwarded local port on the client | Identify which side collides. **On DGX Spark:** run `sudo ss -ltnp 'sport = :8080'` (or `sudo lsof -i :8080`) to find the pre-existing process holding the port; stop that process (recommended — DGX Spark resources are limited, so avoid a competing service) or, if it must keep running, change the published port in `deploy/compose/compose.yaml`. **On the client:** stop the process using the local port and re-forward, or forward to a free local port (for example `ssh -L 8081:localhost:8080 <user>@<host>`) and update the MCP client URL to match (`http://localhost:8081/mcp/cuda-docs`). |
