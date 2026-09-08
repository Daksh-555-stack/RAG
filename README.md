# 📄 IntelliDocs AI

**A Retrieval-Augmented Generation (RAG) chatbot that answers questions strictly from your uploaded PDFs — with sources shown for transparency.**

Upload one or more PDF documents, and IntelliDocs AI extracts the text, chunks it, converts it into vector embeddings, and stores it in a local FAISS index. You can then chat with your documents: every answer is generated **only** from the retrieved chunks of your files (no general model knowledge), and each reply quotes the exact source chunk — filename, page, and snippet — it was grounded on.

The whole pipeline runs **locally and free**: the embedding model (`all-MiniLM-L6-v2`) needs no API key and works offline, and the default answer-generation model uses the free Hugging Face Inference API (a free token is all you need). OpenAI and Groq are supported as optional providers.

---

## ✨ What this project demonstrates (resume bullets)

- Built an end-to-end **Retrieval-Augmented Generation (RAG)** system in Python: PDF ingestion → text extraction → chunking → embeddings → vector search → LLM-grounded answer generation.
- Engineered a **modular, production-quality pipeline** (`src/` package) with per-module responsibilities, docstrings, and defensive error handling around every external call (PDF parsing, model loading, API inference).
- Integrated the modern **LangChain package split** (`langchain`, `langchain-community`, `langchain-huggingface`, `langchain-openai`, `langchain-groq`) with `RetrievalQA`, `RecursiveCharacterTextSplitter`, and FAISS vector stores.
- Designed a **provider-agnostic LLM layer** — free Hugging Face Inference API by default, with pluggable OpenAI/Groq backends — plus a documented fully-local transformers alternative for zero-API operation.
- Implemented **document fingerprinting and index persistence** (SHA-256 of file sets → cached FAISS stores) so re-uploading identical documents skips re-embedding, and added **source citation** (filename + page + quoted snippet) for answer transparency.
- Built a polished **Streamlit UI** with session-based chat memory, cached model loading, progress states, and graceful failure handling.

---

## 🏗️ Architecture

```
                        ┌─────────────────────────────────────────────────┐
                        │                  Streamlit app.py                │
                        │                                                 │
  ┌──────────┐   upload  │  ┌──────────────┐    ┌──────────────────────┐  │
  │  User    │ ────────► │  │  Sidebar     │    │  Chat interface      │  │
  │ (browser)│           │  │  (PDFs +     │    │  (questions +        │  │
  └────▲─────┘           │  │  LLM config) │    │  answers + sources)  │  │
       │                 │  └──────┬───────┘    └──────────▲───────────┘  │
       │   rendered      │         │                        │             │
       └─────────────────│─────────┼────────────────────────┘             │
                        └─────────┼──────────────────────────────────────┘
                                  │
              ┌───────────────────┴────────────────────────────────────┐
              │                    RAG pipeline (src/)                  │
              │                                                         │
              │  PDF Upload ──► Text Extraction ──► Chunking            │
              │   (pypdf)          (per page)        (1000 chars,       │
              │                                   150 overlap)          │
              │                                                         │
              │  Embeddings ◄── Chunks (with source/page metadata)      │
              │  all-MiniLM-L6-v2 (local, free)                         │
              │         │                                               │
              │         ▼                                               │
              │  FAISS Vector Store  ◄──── saved to data/vector_store/  │
              │         │            (reused via file-hash fingerprint) │
              │         ▼                                               │
              │  Retriever (top-k = 4)                                  │
              │         │                                               │
              │         ▼                                               │
              │  Prompt + Context ──► LLM (HF Inference API / OpenAI /  │
              │                       Groq) ──► Grounded Answer + Sources│
              └─────────────────────────────────────────────────────────┘
```

## 📁 Project structure

```
intellidocs-ai/
│
├── app.py                     # Streamlit main app (UI + chat loop)
├── requirements.txt
├── .env.example                # Template for API keys
├── .gitignore
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── pdf_loader.py           # Extract text from uploaded PDFs (pypdf)
│   ├── text_splitter.py        # Chunk text into overlapping segments
│   ├── embeddings.py           # Load HuggingFace embedding model (cached)
│   ├── vector_store.py         # Build/save/load FAISS index
│   ├── qa_chain.py             # RetrievalQA chain (retriever + LLM + prompt)
│   └── utils.py                # Helpers (file hashing, source formatting)
│
├── data/
│   └── vector_store/           # Persisted FAISS index files (auto-created)
│
└── assets/
    └── logo.png (optional)     # Shown in the sidebar if present
```

---

## 🚀 Setup

### Prerequisites

- **Python 3.10+** (3.11 or 3.12 recommended; this project was verified on 3.14)
- A free [Hugging Face token](https://huggingface.co/settings/tokens) for the default LLM provider *(optional if you use OpenAI/Groq, or the local transformers alternative)*

### 1. Create a virtual environment and install dependencies

```bash
cd intellidocs-ai
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> 💡 The first run downloads the embedding model (~80 MB) and, if you use the local alternative, the LLM weights. Both are cached on disk afterwards.

### 2. Configure secrets

```bash
cp .env.example .env
```

Open `.env` and paste your Hugging Face token:

```
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
```

(You can also paste keys directly in the app's sidebar — `.env` is only a convenience default.)

### 3. Run the app

```bash
streamlit run app.py
```

Your browser opens `http://localhost:8501`.

---

## 🧪 How to test it

1. **Grab a sample PDF** — download any text-based PDF, e.g. a public research paper or the [Python documentation PDF](https://docs.python.org/3/download.html).
2. **Upload** it in the sidebar (multiple files allowed) and click **🚀 Process documents**. You'll see the chunk count and file list appear under *Index status*.
3. **Ask a grounded question**, e.g. for the Python docs: *"What is a list comprehension and how is it written?"*
4. **Check the sources** — every answer has an expandable **📎 Sources** block showing the filename, page number, and the exact chunk used.
5. **Watch it refuse to hallucinate** — ask something unrelated to the document (e.g. *"What is the capital of France?"*) and the model should reply that it couldn't find the information in the document.

**Re-upload test:** process the same file, clear the session, upload it again, and note it loads the cached index instantly instead of re-embedding.

---

## ⚙️ Configuration

| Setting | Where | Default |
|---|---|---|
| Embedding model | `src/embeddings.py` | `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| Chunk size / overlap | `src/text_splitter.py` | `1000` / `150` characters |
| Retrieved chunks per question (k) | `src/qa_chain.py` | `4` |
| Answer-generation LLM | Sidebar → *Choose the LLM* | `mistralai/Mistral-7B-Instruct-v0.2` via free HF Inference API |
| Persistence dir | `src/vector_store.py` | `data/vector_store/` |

**Switching providers:** pick **OpenAI** (e.g. `gpt-4o-mini`) or **Groq** (fast, free-tier Llama models) in the sidebar and paste the matching key.

**Going fully offline:** see the commented block at the bottom of `src/qa_chain.py` for a local `transformers` pipeline (`HuggingFacePipeline`) that requires no API call at all.

---

## ☁️ Deployment (free)

### Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app** → pick the repo and set the main file to `app.py`.
3. In **Settings → Secrets**, add your keys:
   ```toml
   HUGGINGFACEHUB_API_TOKEN = "hf_..."
   # OPENAI_API_KEY = "sk-..."
   # GROQ_API_KEY = "g8_..."
   ```
4. Deploy. Note: the free tier rebuilds the vector store per session unless you point `data/vector_store/` at persistent storage (e.g. an S3 bucket or database).

### Hugging Face Spaces (Streamlit SDK)

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space), choose the **Streamlit** SDK and a CPU basic hardware tier.
2. `git clone` the Space, copy the project files in, and commit/push.
3. Add your token in **Settings → Variables and secrets** as `HUGGINGFACEHUB_API_TOKEN`.

---

## 📸 Screenshots

*(placeholder — add screenshots of the upload flow, chat view, and Sources expander here)*

---

## 🔮 Future improvements

- **More vector databases** — swap in Chroma or Pinecone via a thin store interface (`vector_store.py` is already isolated for this).
- **Chat memory** — feed previous turns into the prompt for follow-up questions (`ConversationBufferMemory`).
- **More file types** — DOCX, TXT, Markdown, and HTML loaders alongside PDF.
- **Authentication & multi-user** — per-user stores and login on Streamlit Cloud.
- **Streaming answers** — stream LLM tokens into the chat UI for a snappier feel.
- **Better retrieval** — hybrid search (BM25 + dense), rerankers, and query rewriting.
- **Scanned PDFs** — OCR via `pytesseract` to unlock image-only documents.

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `Failed to load the embedding model` | Check internet on first run (model downloads once), then it's offline-capable. |
| `401 / model is gated` on the HF provider | Some models require accepting terms on their Hugging Face page; pick another model in the sidebar. |
| Legacy `RetrievalQA` crashes on Python 3.14 | This project deliberately uses the LCEL `create_retrieval_chain` path — the legacy `langchain.chains.base.Chain` machinery hits a pydantic forward-ref bug on 3.14. |
| `inference_api_url` errors | Confirm `langchain-huggingface` is `>=0.2.0` (`pip show langchain-huggingface`); `HuggingFaceEndpoint` is the current class. |
| Slow first answer | The free HF Inference API cold-starts per model; subsequent calls are faster. Use Groq for speed. |
| No text extracted from a PDF | It's a scanned/image-only PDF — upload a text-based PDF (OCR is a planned improvement). |

---

Built with Streamlit, LangChain, sentence-transformers, FAISS, and pypdf. 💙
