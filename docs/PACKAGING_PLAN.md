# Plan to Package and Run `assembly-agent` on GDC Software-Only

This document outlines the step-by-step plan to package the **Assembly Agent** (Frontend + FastAPI Backend + Local Gemma Models + ChromaDB) into a one-action, offline-ready deployment on **Google Distributed Cloud (GDC) Software-Only (Bare Metal)**.

---

## 1. Design Goals & Constraints

1. **Four Local Components on One Machine:**
   - **Frontend:** Static web UI served locally (no external CDN scripts, fonts, or icons).
   - **Backend ([`main.py`](main.py)):** FastAPI + LangGraph agent service.
   - **Local Models:** `google/gemma-4-E2B-it` (generation/vision) and `google/embeddinggemma-300m` (embeddings).
   - **Vector Store:** Local ChromaDB seeded with [`brimes-assembly.pdf`](brimes-assembly.pdf).
2. **Cost-Effective, Portable Hardware (≤ £500):**
   - Must run smoothly on a lightweight, backpack-portable Mini-PC (CPU-only, no discrete GPU required).
3. **Online Install, 100% Offline Operation:**
   - Initial setup uses the internet to pull container images and pre-packaged model weights.
   - At the event venue, the PC operates with zero internet access (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `imagePullPolicy: IfNotPresent`).
4. **Zero-Touch Onsite Experience (No USBs, No CLI for Marketers):**
   - Developers publish images and model weights to Google Cloud once.
   - Onsite staff run a single installer script while on Wi-Fi, and then launch the app via a desktop icon (`http://localhost:8080`).

---

## 2. Target Hardware & Resource Budget

### Recommended Event PC Spec (≤ £500 Mini-PC Tier)
- **Model Class:** Beelink SER / Minisforum / Intel NUC (approx. 0.6 kg).
- **CPU:** 8-core / 16-thread `x86-64-v3` processor (e.g., AMD Ryzen 7 5800H / 7735HS or Intel Core i5/i7 12th/13th Gen).
  - *Note:* GDC Bare Metal strictly requires **`x86-64-v3`** CPU microarchitecture or higher.
- **RAM:** **32 GB DDR4/DDR5** (16 GB strict minimum enforced by pre-flight checks).
- **Disk:** **512 GB NVMe SSD** (GDC requires a minimum of **128 GiB** free storage).
- **OS:** Ubuntu 22.04 LTS (certified vendor kernel with `cgroupsv2` enabled and `ufw` disabled).

### Memory & CPU Allocation on the Node
By deploying GDC Software-Only as a **`standalone` cluster** with **`profile: edge`**, control-plane overhead is reduced to ~6 GB RAM:

| Component | CPU Request / Limit | RAM Request / Limit | Disk / Volume |
|---|---|---|---|
| **GDC Edge Control Plane + System Pods** | 2 vCPUs | ~6 GiB | ~30 GiB host disk |
| **Backend + Local Models (`gemma-4-E2B-it` + `embeddinggemma-300m`)** | `2000m` / `6000m` | `6Gi` / `10Gi` | `20Gi` local PV (`/models`, ReadOnly) + `5Gi` (`/uploads`) |
| **ChromaDB (`chromadb/chroma:0.6.3`)** | `250m` / `1000m` | `512Mi` / `1536Mi` | `5Gi` local PV (`/chroma/chroma`, ReadWriteOnce) |
| **Frontend (`nginx:1.27.3-alpine`)** | `100m` / `500m` | `64Mi` / `128Mi` | Stateless |
| **Total Footprint** | **~4.5–8 vCPUs** | **~13–17.5 GiB RAM** | **~60 GiB used** (fits comfortably in 32 GB RAM / 512 GB SSD) |

---

## 3. Step-by-Step Packaging & Implementation Plan

### Phase 1: Containerization & Offline Model Staging
1. **Pin All Dependencies:**
   - Pin exact package versions in [`requirements.txt`](requirements.txt) and exact container base image tags (`python:3.11.11-slim-bookworm`, `nginx:1.27.3-alpine`, `chromadb/chroma:0.6.3`).
2. **Decouple Model Weights from the Backend Image:**
   - Instead of baking 5+ GB of weights into the backend Docker image or downloading from Hugging Face at runtime, stage `google/gemma-4-E2B-it` and `google/embeddinggemma-300m` once (at a pinned revision SHA) into a versioned `.tar.zst` archive stored in a private Google Cloud Storage (GCS) bucket.
   - During installation on the event PC, unpack the weights onto the host at `/var/lib/assembly-agent/models/` and mount that directory into the backend pod at `/models`.
   - Set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_HUB_DISABLE_TELEMETRY=1` in the backend container so it loads exclusively from `/models/...` and never calls `huggingface.co`.
3. **Add Kubernetes Health & Readiness Endpoints to [`main.py`](main.py):**
   - `GET /healthz`: Returns `200 OK` as soon as the FastAPI process is alive.
   - `GET /readyz`: Returns `200 OK` only after the Gemma model is loaded in memory and ChromaDB responds with a non-zero document count.
4. **Make ChromaDB Seeding Idempotent ([`document_embeddings.py`](document_embeddings.py)):**
   - Check `vectorstore._collection.count()` before embedding [`brimes-assembly.pdf`](brimes-assembly.pdf) so pod restarts do not re-embed and duplicate chunks.
   - Set `ANONYMIZED_TELEMETRY=FALSE` on both ChromaDB and the backend client.
5. **Frontend Container (`images/frontend/`):**
   - Package the web UI into an Nginx container that serves static assets on `/` (with all fonts/icons bundled locally) and reverse-proxies `/api/*` to the backend service (`http://backend:8000`).

---

### Phase 2: Kubernetes Helm Chart (`deploy/chart/`)
Create a Helm chart with `values.yaml` and `values-offline.yaml`:
1. **Workloads:**
   - `frontend`: `Deployment` (1 replica) exposed via `NodePort` (`30080`) / `hostPort` (`8080`) bound to `127.0.0.1`.
   - `backend`: `Deployment` (1 replica) mounting `model-weights-pvc` (`/models`) and `uploads-pvc` (`/uploads`).
   - `chromadb`: `StatefulSet` (`replicas: 1`—ChromaDB only supports a single writer) mounting `chroma-data-pvc` (`/chroma/chroma`).
2. **Offline Pod Policies:**
   - Set `imagePullPolicy: IfNotPresent` on every container so restarts without internet never trigger `ImagePullBackOff`.
   - Configure lenient startup/readiness probes on `backend` (`initialDelaySeconds: 30`, `periodSeconds: 10`, `failureThreshold: 30`) so Kubernetes waits up to 5 minutes for CPU model loading without restarting the pod.
   - Keep GPU resource limits (`nvidia.com/gpu`) disabled by default (`gpu.enabled: false`) with a toggle in `values.yaml` if GPU-equipped PCs are ever used.

---

### Phase 3: GDC Single-Node Cluster Bootstrap & One-Action Installer (`installer/`)
Build `installer/install.sh` to run once while the PC is online before an event:
1. **Stage 1 — Pre-flight Checks (`installer/preflight.sh`):**
   - Verify Ubuntu LTS version, `x86-64-v3` CPU support, ≥ 16 GB RAM, ≥ 128 GB free disk, `cgroupsv2` enabled, `CONFIG_BPF_JIT=y`, `ufw` disabled, system clock sanity, and reachability to Google Cloud APIs/GCS. Refuse early with plain-English messages if any check fails.
2. **Stage 2 — Install Pinned Tooling:**
   - Install `docker`/`containerd`, `bmctl`, `gcloud`, `kubectl`, and `helm` at the exact versions pinned in `installer/versions.env`.
3. **Stage 3 & 4 — Stable Local Network Identity & Cluster Creation:**
   - Create a persistent Linux dummy interface (`dummy0` with static IP `10.254.254.1/24` via `systemd-networkd` / `netplan`) and bind the GDC `standalone` (`profile: edge`) cluster node IP and MetalLB VIP pool (`10.254.254.100-10.254.254.110`) to `dummy0`.
   - **Why this matters:** If the cluster binds to a hotel/office DHCP IP (`eth0`/`wlan0`), unplugging the cable or joining venue Wi-Fi changes the IP and wedges `etcd` and `kubelet`. Binding to `dummy0` makes the cluster completely immune to network changes.
   - Run `bmctl create cluster -c assembly-edge` and remove the control-plane taint (`kubectl taint nodes --all node-role.kubernetes.io/master:NoSchedule-`) so application pods schedule on the single node.
4. **Stage 5 & 6 — Zero-Touch Artifact Pull & Credential Cleanup:**
   - Use a minimal-permission, read-only bootstrap credential (`roles/artifactregistry.reader`, `roles/storage.objectViewer`) to pull the container images into the node's `containerd` store and download/verify (SHA-256) the model weights from GCS into `/var/lib/assembly-agent/models/`.
   - Shred/remove the bootstrap pull credential once artifacts are verified on disk.
5. **Stage 7 & 8 — Helm Deploy & Automated Self-Test:**
   - Deploy the chart with `helm upgrade --install assembly-agent ./deploy/chart -f ./deploy/chart/values-offline.yaml`.
   - Run an end-to-end self-test: verify `/readyz` is healthy, ChromaDB has the expected chunk count for `brimes-assembly.pdf`, and a test `/call_agent` prompt returns a valid response with `HF_HUB_OFFLINE=1`.
6. **Stage 9 — Desktop Shortcuts for Non-Technical Staff:**
   - Create an **"Assembly Assistant"** desktop launcher (`installer/launcher.sh`) that shows a friendly *"Starting up, this can take a couple of minutes..."* local splash screen while polling `/readyz`, then automatically opens `http://localhost:8080`.
   - Create a **"Create Diagnostics Bundle"** desktop shortcut (`installer/diagnose.sh`) that packages pod logs, node status, disk usage, and installer logs into a single redacted `.zip` file on the Desktop if onsite staff ever need support.

---

### Phase 4: Offline Hardening & Verification Matrix (`tests/offline/`)
Before shipping event PCs, execute and record the 8-point offline verification checklist in `docs/VERIFICATION.md`:
1. **Fresh Online Install:** `install.sh` completes with exit code `0` and self-test passes.
2. **Cold Offline Reboot:** Disable all network interfaces, reboot the Mini-PC, click the desktop icon, and confirm the cluster and `http://localhost:8080` recover automatically.
3. **Offline End-to-End Flow:** Start a session, upload a cabinet photo, ask an assembly question, verify ChromaDB retrieval and Gemma response, and end the session.
4. **Subnet Hopping / Wi-Fi Toggling:** Switch between networks and airplane mode while the app is running; confirm the `dummy0`-backed cluster stays healthy.
5. **Pod Self-Healing Offline:** Delete the `backend` and `chromadb` pods while offline; confirm Kubernetes recreates them from local cached images and ChromaDB data persists.
6. **Idempotent Re-Install:** Re-run `install.sh` over an existing installation; confirm it upgrades cleanly in place without duplicating ChromaDB chunks.
7. **Zero Outbound Traffic Audit:** Monitor DNS/socket connections (`tcpdump` / `ss`) during an offline session to verify zero outbound connection attempts.
8. **Clock & Certificate Check:** Verify cluster certificate expiration dates (`kubeadm certs check-expiration`) before each event batch.
