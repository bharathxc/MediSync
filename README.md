# 🏥 MediSync AI — Unified Healthcare Platform

**A unified portfolio project combining Agentic RAG Healthcare Navigator + Clinical NLP Safeguards with Dual Login (Member & Employee)**

> ⚠️ **Disclaimer:** This is a portfolio/educational project — not for clinical use.

---

## 📋 Table of Contents

1. [Dual Login System](#-dual-login-system)
2. [Project Overview](#-project-overview)
3. [Architecture](#-architecture)
4. [Technologies & Stack](#-technologies--stack)
5. [Core Concepts Explained](#-core-concepts-explained)
6. [Components](#-components)
7. [Interview-Ready Explanations](#-interview-ready-explanations)
8. [Setup & Running](#-setup--running)
9. [Employee Credentials](#-employee-credentials)
10. [Future Improvements](#-future-improvements)

---

## 🔐 Dual Login System

MediSync AI features a **dual login system** that provides differentiated experiences for Members vs. Employees:

### 👤 Member Portal
- **Access:** Select member profile from sidebar (simulates member login)
- **Features:**
  - Chat with AI benefits navigator
  - Personalized plan details (deductible, copay, PCP)
  - Benefits Q&A (drug coverage, prior auth, mental health)
- **Goal:** Self-service benefits understanding

### 👨‍💼 Employee Portal (HR/Provider)
- **Access:** Email/password login (see credentials below)
- **Features:**
  - **Member Support Tab** — Full CRM lookup + chat as support agent
  - **Clinical NLP Safeguard Tab** — De-identification + Bias Auditing
  - **Analytics Tab** — Member statistics dashboard
- **Goal:** Employee assistance and admin tools

### Login Flow
```
┌────────────────┐
│  🏥 MediSync AI │
│   Login Page    │
└───────┬────────┘
        │
   ┌────┴────┐
   ▼         ▼
👤 Member  👨‍💼 Employee
   │         │
   ▼         ▼
┌─────────┐ ┌─────────────┐
│ Benefits│ │  Employee   │
│Navigator│ │  Portal    │
└─────────┘ └─────────────┘
```

---

## 🏗️ Project Overview

MediSync AI combines two healthcare AI systems into one unified platform:

### 1. Benefits Navigator (Agentic RAG)
An intelligent healthcare benefits assistant that helps users understand their Optum health plan coverage, prescription drugs, copays, and deductibles through:
- **LangGraph orchestration** for multi-step agentic workflows
- **ChromaDB vector search** for semantic policy lookup
- **PII masking** for privacy protection

### 2. Clinical NLP Safeguard
A hybrid de-identification engine for clinical notes that addresses:
- **False positive redaction** (accidentally removing medical terminology like "Down's syndrome")
- **Algorithmic bias** in Named Entity Recognition (NER) for minority demographics
- **Hybrid NLP approach**: Traditional ML + GenAI refinement

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     STREAMLIT UI                               │
│  ┌─────────────────────┐  ┌──────────────────────────────────┐ │
│  │  Benefits Navigator │  │   Clinical NLP Safeguard          │ │
│  │  (Agentic RAG)    │  │  (De-identification + Bias Audit)  │ │
│  └─────────┬──────────┘  └───────────────┬──────────────────────┘ │
│            │                           │                        │
│  ┌────────▼───────────────────────────▼──────────────────┐   │
│  │                    LLM LAYER                            │   │
│  │         (Ollama / Google Gemini)                         │   │
│  └─────────────────────┬───────────────────────────────────┘   │
│                       │                                       │
│  ┌───────────────────▼───────────────────────────────────┐    │
│  │                  TOOLS & HELPERS                      │    │
│  │  - RAG (ChromaDB)  - CRM Lookup  - Guardrails      │    │
│  │  - Presidio NER    - Bias Auditor                 │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Technologies & Stack

| Category | Technology | Purpose |
|----------|-----------|--------|
| **UI** | Streamlit | Web interface |
| **LLM Orchestration** | LangChain, LangGraph | Agent state machine |
| **Vector Database** | ChromaDB | Semantic search for RAG |
| **NLP/De-ID** | Presidio, spaCy | Named Entity Recognition |
| **LLM Backend** | Ollama (local) / Google Gemini | Language models |
| **PDF Processing** | pypdf | Knowledge base ingestion |
| **Visualization** | Plotly | Bias audit charts |

---

## 📚 Core Concepts Explained

### 🔹 1. Retrieval-Augmented Generation (RAG)

**What it is:**
RAG combines a retrieval system with a generative model. Instead of relying solely on what an LLM "knows" (which may be outdated or hallucinated), RAG:
1. Retrieves relevant documents from a knowledge base
2. Feeds them as context to the LLM
3. Generates answers grounded in actual documents

**Why it matters for healthcare:**
- Insurance policies change frequently — RAG ensures answers come from current documents
- Reduces LLM hallucination (a critical safety concern in healthcare)
- Enables source attribution ("Based on your Basic Plan document, page 5...")

**In this project:**
- PDFs (Basic/Premium plans, Medicare forms) are chunked and stored in ChromaDB
- User queries are embedded and similarity-searched against the vector store
- Retrieved chunks are passed to the LLM as context

```
User: "How much is insulin copay?"
    │
    ▼
[Embed Query] → "insulin copay cost"
    │
    ▼
[ChromaDB Search] → Top 5 chunks
    │
    ▼
[LLM Prompt] = System + Retrieved Context + User Question
    │
    ▼
[LLM Response]: "On your Basic Plan, insulin is Tier 2..."
```

---

### 🔹 2. LangGraph (Agent Orchestration)

**What it is:**
LangGraph is a library for building stateful, multi-step agent workflows using a directed graph structure. Each node is a function, and edges determine the flow.

**Key concepts:**
- **State**: A TypedDict that flows through all nodes (messages, member_info, etc.)
- **Nodes**: Python functions that transform state (e.g., supervisor_node, tool_executor_node)
- **Conditional Edges**: Routing logic based on state (use tools vs. respond directly)

**Why it matters:**
- Enables complex, multi-step reasoning (not just single prompt → response)
- Supports tool-use planning (the agent decides WHEN to call tools)
- Maintains conversation context across turns

**In this project:**
```
supervisor_node → (decides: tools or respond?)
    │
    ├─[tools]─→ tool_executor_node → response_node → END
    │
    └─[respond]──→ END
```

---

### 🔹 3. Vector Embeddings & ChromaDB

**What are embeddings?**
Embeddings convert text into numerical vectors such that similar texts have similar vectors. "How much is insulin?" and "insul pricing" → close vectors in N-dimensional space.

**Why ChromaDB?**
- Persistent vector store (survives restarts)
-Simple API for similarity search
- Built on ClickHouse for speed

**In this project:**
- PDF pages → chunked → embedded → stored in ChromaDB
- Query → embedded → top-k nearest chunks retrieved

---

### 🔹 4. Named Entity Recognition (NER) with Presidio

**What is NER?**
NER identifies and classifies entities in text (names, dates, locations, medical terms). Used here for PII detection.

**Why Presidio?**
- Microsoft's standardized PII detection framework
- Supports multiple "recognizers" (spaCy, regex, custom)
- Built-in anonymization operators

**In this project:**
- spaCy `en_core_web_lg` model for PERSON, DATE, LOCATION detection
- Custom operators for replacement (e.g., [REDACTED_NAME])

---

### 🔹 5. Hybrid De-identification Pipeline

**The Problem:**
Traditional NER has high recall but low precision — it catches too much:
- "Down's syndrome" → incorrectly flagged as a NAME
- "Mayo Clinic" → incorrectly redacted (important hospital info)

**The Solution:**
Hybrid pipeline combines:
1. **Traditional Pass (Presidio)**: High recall — catches everything that might be PII
2. **GenAI Refinement Pass**: The LLM reviews flagged entities and decides what to actually redact, preserving medical terminology

```python
# Step 1: Presidio flags "Mayo Clinic" as LOCATION → [REDACTED_HOSPITAL]
anonymized = "Patient John Smith was admitted to [REDACTED_HOSPITAL]..."

# Step 2: LLM reviews → "Mayo Clinic is a hospital, don't redact"
final = "Patient John Smith was admitted to Mayo Clinic..."
```

---

### 🔹 6. Bias Auditing in NLP Models

**The Problem:**
NLP models (including spaCy) can systematically underperform on minority demographics due to training data imbalances. A model trained predominantly on Western names may fail to detect:
- African names (Kwame Okonkwo)
- South Asian names (Priya Patel)
- Middle Eastern names (Aisha Mohammed)

**How we measure it:**
- **Detection Rate (Recall)** = True Positives / Total Entities
- Compare across demographic groups
- Flag any group with < 80% detection rate

**Why it matters:**
- HIPAA compliance requires consistent PII removal
- Liability if a patient's name leaks due to model bias
- Ethical AI responsibility

---

### 🔹 7. PII Masking & Guardrails

**What it is:**
PII Masking scans user input for sensitive data (SSN, phone, email) BEFORE it reaches the LLM, replacing with tokens like [REDACTED_SSN].

**Why it matters:**
- **Data minimization**: The LLM never sees raw PII
- **Conversation safety**: Even if a user types "my SSN is 123-45-6789", it's masked
- **Compliance**: Reduces attack surface for prompt injection

**In this project:**
- Regex patterns for SSN, phone, email, DOB, Medicare ID
- Guardrails module also validates LLM outputs for hallucination

---

### 🔹 8. Tool-Using Agents

**What it is:**
An "agent" is an LLM that can call external tools to complete tasks. The LLM decides WHEN to use tools, not just how to respond to text.

**In this project:**
- `get_policy_info`: Searches ChromaDB for benefits info
- `get_member_details`: Looks up CRM member profiles
- The Supervisor node decides whether to call tools based on user intent

---

## 💻 Components

### Agents Module (`agents/`)

| File | Purpose |
|------|---------|
| `agent.py` | LangGraph workflow: Supervisor → Tools → Response |
| `tools.py` | RAG and CRM lookup tools |
| `guardrails.py` | PII masking, response grounding |

### NLP Module (`nlp/`)

| File | Purpose |
|------|---------|
| `hybrid_pipeline.py` | Presidio + GenAI de-identification |
| `bias_auditor.py` | Demographic bias evaluation |

### Root Files

| File | Purpose |
|------|---------|
| `config.py` | Unified configuration |
| `ingest.py` | PDF → ChromaDB pipeline |
| `app.py` | Unified Streamlit UI |

---

## 🚀 Setup & Running

### Prerequisites
- Python 3.10+
- Ollama installed locally (optional, for local LLM)
- spacy model: `python -m spacy download en_core_web_lg`

### Installation

```bash
# Clone and navigate
cd medisync-ai

# Create and activate venv
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Running

```bash
# Start Ollama (if using local model)
ollama serve
ollama pull llama3.2

# Run the app
streamlit run app.py
```

### First-time setup

```bash
# Ingest PDFs to vector store
python ingest.py
```

---

## 🔮 Future Improvements

- [ ] Add real-time document ingestion from CMS/Medicare APIs
- [ ] Implement retrieval citation with exact page references
- [ ] Add multi-language support (Spanish, Mandarin)
- [ ] Integrate with FHIR for actual EHR interoperability
- [ ] Add conversation memory summarization for long chats
- [ ] Implement guardrails for harmful content detection
- [ ] Add LangSmith observability
- [ ] Move to production LLM (OpenAI, Anthropic)

---

## 🔑 Employee Credentials

For testing the **Employee Portal**:

| Email | Password | Role |
|-------|----------|------|
| `admin@optum.com` | `admin123` | HR Administrator |
| `provider@optum.com` | `provider123` | Healthcare Provider |
| `analyst@optum.com` | `analyst123` | Data Analyst |

---

## 📚 Learning Resources

### RAG & Vector Search
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [Vector Embeddings Explained](https://magazine.seaboard.ai/p/e5-embeddings-explained)

### Agentic Systems
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [ReAct: Synergizing Reasoning + Acting](https://arxiv.org/abs/2210.03629)

### NLP & De-identification
- [Microsoft Presidio](https://microsoft.github.io/presidio/)
- [spaCy NER](https://spacy.io/usage/feature-overview)

### Healthcare AI Ethics
- [NIST AI Risk Management Framework](https://airc.nist.gov/AI-RMF)
- [HIPAA Guidelines](https://www.hhs.gov/hipaa/index.html)

---

## 📄 License

MIT — Educational/Portfolio Use Only. Not for clinical deployment.

---
