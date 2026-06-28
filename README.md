# 🤖 Digital Twin AI

> **Personal AI Digital Twin — A conversational RAG assistant that knows everything about Imran**

Digital Twin AI is a Retrieval-Augmented Generation (RAG) assistant built on OpenAI GPT-4o and ChromaDB. It embeds Imran's professional profile, experience, skills, and background into a vector database — then answers any question about him as if it were Imran himself speaking.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=flat&logo=openai&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-blue?style=flat)
![Gradio](https://img.shields.io/badge/Gradio-UI-FF7C00?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 🧠 What is a Digital Twin AI?

A Digital Twin AI is a **personal knowledge assistant** — an AI that represents *you*. Instead of a generic chatbot, it:

- Is trained on your actual professional documents (resume, experience, skills, bio)
- Answers questions in first person, speaking as you
- Retrieves the most relevant context from your knowledge base before responding
- Can be shared with recruiters, colleagues, or anyone who wants to know more about you

---

## ✨ Features

| Feature | Description |
|---|---|
| 🗂️ **RAG Knowledge Base** | Professional profile embedded in ChromaDB vector store |
| 🔍 **Semantic Search** | Questions matched to relevant document chunks via embeddings |
| 💬 **First-Person Persona** | Responds as Imran — professional, authentic, personable |
| 🌊 **Streaming Responses** | GPT-4o streams answers token-by-token |
| 🧠 **Multi-turn Context** | Full conversation history preserved across the session |
| 🎨 **Custom UI** | Gradio chat interface with profile photo and branding |

---

## 🔄 How It Works — RAG Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│  USER: Asks a question about Imran                                   │
│  e.g. "What cloud platforms have you worked with?"                   │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
╔══════════════════════════════════════════════════════════════════════╗
║  STEP 1 — EMBED QUESTION                                             ║
║                                                                      ║
║  OpenAI text-embedding-ada-002                                       ║
║  Question text → dense vector (1536 dimensions)                      ║
╚══════════════════════════════════════════════════════════════════════╝
                               │
                               ▼
╔══════════════════════════════════════════════════════════════════════╗
║  STEP 2 — RETRIEVE CONTEXT                                           ║
║                                                                      ║
║  ChromaDB vector similarity search                                   ║
║  Query vector vs stored document chunk vectors                       ║
║                                                                      ║
║  Knowledge base chunks include:                                      ║
║    - Professional overview and summary                               ║
║    - Work experience (Cotiviti, Genpact, roles, achievements)        ║
║    - Skills (AWS, AI/ML, ITSM, cloud, automation)                   ║
║    - Education (Osmania University, University of Hyderabad)         ║
║    - Personal interests and personality                              ║
║                                                                      ║
║  Returns: top-K most relevant chunks (K=3 by default)               ║
╚══════════════════════════════════════════════════════════════════════╝
                               │
                               ▼
╔══════════════════════════════════════════════════════════════════════╗
║  STEP 3 — GENERATE RESPONSE                                          ║
║                                                                      ║
║  GPT-4o prompt:                                                      ║
║    System: "You are Imran's digital twin. Answer as Imran in         ║
║             first person using only the provided context."           ║
║                                                                      ║
║    Context: [retrieved chunks from ChromaDB]                         ║
║    History: [prior conversation turns]                               ║
║    Question: [user's question]                                       ║
║                                                                      ║
║  GPT-4o streams the response token by token                          ║
╚══════════════════════════════════════════════════════════════════════╝
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Imran's Digital Twin answers — in first person, with full context   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Knowledge Base — What Imran's Twin Knows

| Domain | Content |
|---|---|
| **Professional Role** | Assistant Manager IT Service Desk at Cotiviti (11+ years enterprise IT) |
| **Core Expertise** | AWS cloud, AI/ML (LLMs, RAG, Digital Twin), ITSM, ServiceNow, Jira |
| **Leadership** | Global IT team management, SLA compliance, workforce planning |
| **Technical Skills** | Python, cloud architecture, automation, identity & access management |
| **Education** | BSc Computer Science (Osmania University) · PG Diploma Cyber Crime (UoH) |
| **Certifications** | ITIL-aligned processes, Microsoft 365 admin, Active Directory |
| **Personal** | Cricket and football fan, loves cats and dogs, enjoys travel and new cuisines |

---

## 🏗️ Project Structure

```
Digital_Twin-AI_Project/
│
├── app.py              ← Full RAG pipeline + Gradio chat UI
├── requirements.txt    ← Python dependencies
├── Imran.png           ← Profile photo displayed in chat UI
└── README.md
```

### Inside `app.py`

| Section | Purpose |
|---|---|
| **Setup** | OpenAI client init, ChromaDB persistent client setup |
| **Document Store** | 5 knowledge documents embedded into ChromaDB on startup |
| **RAG Function** | `query_digital_twin(question, history)` — embed → retrieve → generate |
| **UI** | `gr.ChatInterface` with profile image, description, example questions |

---

## 🔧 Architecture Decisions

### Why ChromaDB?

ChromaDB is a lightweight, file-based vector database that runs locally without infrastructure setup. Documents are embedded once on startup and persisted to disk — subsequent runs skip re-embedding if the collection already exists.

### Why RAG instead of fine-tuning?

RAG is more practical for personal knowledge bases:
- **Updateable** — add new documents without retraining
- **Transparent** — you can see exactly which chunks were retrieved
- **Cost-effective** — no fine-tuning compute required
- **Accurate** — grounded in actual documents, not model memory

### First-Person Persona

The system prompt instructs GPT-4o to respond as Imran in first person. This creates a more natural, authentic interaction than a third-person assistant ("Imran has worked at...").

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/emran-Automation-Techlead/Digital_Twin-AI_Project.git
cd Digital_Twin-AI_Project
pip install -r requirements.txt
```

### 2. Configure

Create a `.env` file:
```env
OPENAI_API_KEY=sk-proj-...
```

### 3. Launch

```bash
python app.py
```

The app embeds documents into ChromaDB on first run, then opens the Gradio chat UI.

---

## 💡 Example Questions to Ask

- *"What's your experience with AWS?"*
- *"Tell me about your role at Cotiviti"*
- *"What AI projects have you built?"*
- *"What are your key technical skills?"*
- *"How do you approach IT service management?"*
- *"What are you passionate about outside of work?"*

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **UI** | Gradio ChatInterface |
| **AI** | OpenAI GPT-4o (generation) + text-embedding-ada-002 (embeddings) |
| **Vector DB** | ChromaDB (local persistent storage) |
| **Language** | Python 3.10+ |

---

## 📄 License

MIT — Personal portfolio project.
