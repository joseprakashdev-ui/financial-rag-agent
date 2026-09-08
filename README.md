# Financial RAG Agent — Document Intelligence & Compliance Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.2+-green.svg)](https://python.langchain.com/)
[![VectorDB](https://img.shields.io/badge/VectorDB-ChromaDB%20%7C%20FAISS-orange.svg)](https://docs.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Production-ready Retrieval-Augmented Generation (RAG) pipeline engineered for automated financial compliance auditing, credit risk assessment, and policy document intelligence.

---

## 📌 Overview

**Financial RAG Agent** is an enterprise-grade document intelligence system designed for Banking, Financial Services, and Insurance (BFSI) workflows. It ingests complex regulatory guidelines, commercial credit policies, loan compliance covenants, and earnings reports, allowing risk analysts and compliance officers to perform precise, natural-language semantic queries backed by verifiable source citations.

By combining context-aware chunking, dense vector embeddings, and LangChain orchestration with LLMs (Google Gemini / Anthropic / OpenAI / Bedrock), this engine eliminates hallucinations and ensures audit-ready compliance verification.

---

## 🚀 Key Features

- 📄 **Multi-Format Document Ingestion**: Ingests unstructured financial policies, PDFs, DOCX, and regulatory bulletins.
- 🧩 **Context-Aware Semantic Chunking**: Employs recursive boundary-aware text splitting preserving financial tables, numerical covenants, and hierarchical sections.
- 🔍 **Hybrid Dense Vector Retrieval**: Embedded with ChromaDB / FAISS leveraging high-dimensional semantic search and cosine similarity ranking.
- 🤖 **LLM Orchestration with LCEL**: Built using LangChain Expression Language (LCEL) with strict grounding prompt templates to mitigate risk and enforce zero-hallucination policies.
- 📑 **Source Attribution & Citations**: Every generated finding is annotated with exact document metadata, page numbers, and chunk excerpts for auditability.
- 🛡️ **BFSI Compliance Ready**: Pre-configured for commercial lending criteria, Debt Service Coverage Ratios (DSCR), Loan-to-Value (LTV) limits, and collateral validation.

---

## 🏗️ Architecture Flow

```
┌────────────────────────┐
│ Financial Policy Docs  │ (Credit Guidelines, SEC 10-K, LTV Covenants)
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ Recursive Chunker      │ (Preserves tables, numeric clauses & section context)
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ Embedding Model        │ (HuggingFace sentence-transformers / OpenAI embeddings)
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ Vector Database        │ (ChromaDB / FAISS vector indexing)
└───────────┬────────────┘
            │
       User Query: "What is the maximum LTV and minimum DSCR for commercial real estate?"
            │
            ▼
┌────────────────────────┐
│ Semantic Retriever     │ (Top-k similarity retrieval with relevance score)
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ Grounded LLM Chain     │ (LangChain LCEL Prompt + Gemini / Claude / GPT-4)
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ Verifiable Answer      │ (Precise findings + direct document source citation)
└────────────────────────┘
```

---

## 🛠️ Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/joseprakashdev-ui/financial-rag-agent.git
cd financial-rag-agent
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
# or OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Run the RAG Pipeline
```bash
python rag_engine.py
```

---

## 📊 Sample Query & Response Execution

**Query:**
> *"What are the mandatory Loan-to-Value (LTV) limits and DSCR requirements for Tier-1 Commercial Real Estate loans?"*

**Generated Response:**
```
[COMPLIANCE CHECK: VERIFIED]
Based on Section 4.2 (Credit Underwriting Standards) of the Commercial Credit Risk Policy:
- Maximum Loan-to-Value (LTV): 75.0% for stabilized commercial properties.
- Minimum Debt Service Coverage Ratio (DSCR): 1.25x based on trailing twelve months (TTM) net operating income.
- Exception Protocol: Any LTV > 75% requires Tier-2 Credit Committee escalation and 120% collateral coverage.

Sources Cited:
- Document: sample_policy.txt (Section 4.2 - CRE Lending Standards, Chunk #2)
- Relevance Score: 0.924
```

---

## 📁 Repository Structure

```
financial-rag-agent/
├── README.md               # Comprehensive architectural & usage documentation
├── requirements.txt        # Production dependency specifications
├── rag_engine.py           # Core RAG retrieval, vector store, and LCEL chain
└── sample_data/
    └── sample_policy.txt   # Commercial credit risk & loan compliance policy
```

---

## 👨‍💻 Author

**Jose Prakash T**
- Generative AI Engineer | RAG Architect | BFSI Domain Intelligence
- Focus: LLM Orchestration, Vector Search, Agentic Workflows & Financial Compliance

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
