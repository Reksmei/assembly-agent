# Assembly Agent🛠️

![Assembly Agent Architecture](https://storage.googleapis.com/ai-pg-demo-media/assembly_agent_architecture.png)

An offline-capable, multimodal Retrieval-Augmented Generation (RAG) agent with voice output, built with **FastAPI**, **LangChain / LangGraph**, **ChromaDB**, **Google Gemma** (`gemma-4-E2B-it` + `embeddinggemma-300m`), and **Hugging Face TTS** (`facebook/mms-tts-eng`).

The agent helps users assemble an IKEA BRIMNES cabinet step by step. Users can ask natural-language questions and upload photos of their assembly progress; the agent queries the embedded PDF instruction manual, inspects the uploaded image, and returns both text guidance and spoken audio (`WAV`) so users can listen hands-free while assembling.

Whilst built in this case for assembling IKEA furniture, this architecture can be easily repurposed for motorsport garages to help engineers repair/build equipment, toystores to help children build lego, and much more, all whilst avoiding sharing data with the cloud.

***Note the frontend has yet to be built***

---

## How It Works

1. **Document Ingestion ([`document_embeddings.py`](document_embeddings.py)):**
   Loads [`brimes-assembly.pdf`](brimes-assembly.pdf) using `PyPDFLoader`, splits pages into overlapping text chunks (`chunk_size=1000`, `chunk_overlap=200`) with `RecursiveCharacterTextSplitter`, embeds them locally using `google/embeddinggemma-300m` (`HuggingFaceEmbeddings`), and persists the vectors into `./vectorDB` via ChromaDB.
2. **Retrieval Tool ([`tool.py`](tool.py)):**
   Defines the [`retrieve_relevant_info`](tool.py#L21) LangChain `@tool`, which runs a similarity search (`k=3`) against the persisted ChromaDB store and returns the matching instruction manual text and page numbers.
3. **Stateful Agent, TTS Voice Output & REST API ([`main.py`](main.py)):**
   Initializes the local Gemma chat model (`gemma4:e2b` via `ChatOllama` or `google/gemma-4-E2B-it` via Hugging Face) and binds [`retrieve_relevant_info`](tool.py#L21) inside a LangGraph agent backed by `InMemorySaver`. Each user session is tracked via `thread_id: session_id` so multi-turn follow-up questions retain context. Before returning, the agent's text response is passed through a lightweight Hugging Face `text-to-speech` pipeline (`facebook/mms-tts-eng`) and encoded as base64 WAV audio (`audio_base64`) alongside the text.

---

## Project Structure

```text
assembly-agent/
├── main.py                  # FastAPI server, session endpoints, image upload, and LangGraph agent
├── document_embeddings.py   # PDF loader, text chunker, and ChromaDB embedding ingestion script
├── tool.py                  # LangChain @tool for querying the persisted ChromaDB vector store
├── brimes-assembly.pdf      # Source IKEA BRIMNES cabinet assembly instruction manual
├── vectorDB/                # Local persisted ChromaDB directory
├── embed_docs.sh            # Helper script to run document_embeddings.py standalone
├── requirements.txt         # Python package dependencies
├── Dockerfile               # Container build definition for the backend service
├── .env                     # Local environment variables (e.g., HF_TOKEN, OLLAMA_BASE_URL)
└── docs/
    └── PACKAGING_PLAN.md    # Plan to package and run the full stack offline on GDC Software-Only
```

---

## API Endpoints

| Method | Path | Parameters | Description |
|---|---|---|---|
| `POST` | `/start_session` | None | Generates a `session_id` and creates `uploads-<session_id>/`. |
| `POST` | `/upload_image` | `session_id` (query), `file` (multipart) | Validates image type (`png`, `jpg`, `gif`, `webp`), saves to disk, and returns its URL. |
| `POST` | `/call_agent` | `session_id`, `text_query`, `image_url` (optional) | Invokes the Gemma agent with conversation memory and optional base64-encoded image, and returns both `text` and spoken `audio_base64` (`WAV`) generated via Hugging Face TTS. |
| `DELETE` | `/end_session` | `session_id` | Deletes the `uploads-<session_id>/` directory when the user finishes. |

---

## Running Locally for Development

1. **Create a .env file and set up environment variables:**
   Inside the assembly agent directory, use the command line to create a .env file:
   ```bash
   touch .env
   ```

   Add your Hugging Face token (after accepting the license on `huggingface.co/google/embeddinggemma-300m`) to `.env`
   ```env
   HF_TOKEN=hf_your_token_here
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create a ChromaDB directory**
   ```bash
   mkdir vectorDB
   ```

4. **Pull the local Gemma model in Ollama:**
   ```bash
   ollama pull gemma4:e2b
   ```

5. **Start the server:**
   ```bash
   python3 main.py
   ```
   The API will be available at `http://localhost:8000` (interactive Swagger docs at `http://localhost:8000/docs`).

---

## Running on Google Distributed Cloud (GDC) Software-Only

### Why Run This App on GDC Software-Only?
- **100% Offline Reliability at Events & Venues:** Event floors, pop-up booths, and workshops frequently suffer from congested or non-existent Wi-Fi. Running the frontend, FastAPI backend, ChromaDB, and Gemma models locally on the machine eliminates cloud API latency and internet dependency at runtime.
- **Data Privacy for User Photos:** Assembly photos taken by users never leave the local PC and are deleted automatically when `/end_session` is called.
- **Repeatable Fleet Management & Self-Healing:** Instead of manually configuring Python, Ollama/Hugging Face caches, and systemd services across dozens of event PCs, GDC Software-Only provides a standardized Kubernetes runtime that automatically restarts crashed pods, manages persistent storage volumes, and integrates with Google Cloud Fleet Management when the PCs are back online.

### How It Runs on a Single GDC PC
Because running a full stack (Frontend + FastAPI Backend + ChromaDB + `gemma-4-E2B-it` + `embeddinggemma-300m`) on one portable machine is resource-intensive, the deployment uses:
1. **GDC Bare Metal `standalone` cluster with the `edge` profile (`bmctl`):** Strips Kubernetes control-plane overhead down to ~5–6 GB RAM and 2 vCPUs so all components fit comfortably on a portable, ≤£500 Mini-PC (8-core CPU, 32 GB RAM, 512 GB NVMe SSD).
2. **Pre-staged Local PersistentVolumes:** Model weights (`/models`) and ChromaDB vectors (`/chroma/chroma`) are downloaded once during online provisioning and mounted from local disk with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `imagePullPolicy: IfNotPresent`.
3. **Stable Localhost Entrypoint:** The cluster binds to a persistent local dummy interface (`dummy0`) so venue network changes never disrupt Kubernetes, exposing the web UI at a single desktop shortcut (`http://localhost:8080`).

See **[`docs/PACKAGING_PLAN.md`](docs/PACKAGING_PLAN.md)** for the full architecture, hardware sizing, and step-by-step plan to package and deploy this application on GDC Software-Only.
