# Research Agent with Dual RAG - Technical Documentation

**Project**: Multi-Step Research Agent with Document & Chat History RAG
**Status**: Phase 5 Complete (Authentication + Multi-User Workstation)
**Updated**: December 17, 2024

---

## 🚀 What's New: Dual RAG Capabilities

This research agent now supports **two types of Retrieval-Augmented Generation (RAG)**:

1. **Document RAG**: Upload PDFs, DOCX, TXT → Semantic search retrieves relevant chunks
2. **Chat History RAG**: Past conversations stored → Retrieved for context in new queries

### Key Features

✅ **Pinecone Vector Database** - Cloud-managed, 1536-dim embeddings  
✅ **User Isolation** - Namespace-based separation  
✅ **Galileo Evaluation** - All RAG operations logged  
✅ **Hybrid Ranking** - Combines web + documents + chat  
✅ **Full Authentication** - Clerk-managed multi-user login with backend service token  
✅ **PostgreSQL/SQLite** - Flexible database backend  

---

## Enhanced Workflow (6 Steps + RAG)

```
User Question + RAG Config
    ↓
[PLAN] → Generate research strategy
    ↓
[RAG RETRIEVE] → Query Pinecone for documents + chat history
    ↓
[SEARCH] → Tavily web search
    ↓
[CURATE] → Merge & rank: web + documents + chat (hybrid ranking)
    ↓
[ANALYZE] → Extract insights from all sources
    ↓
[SYNTHESIZE] → Create final answer with RAG context
    ↓
[VALIDATE] → Check groundedness
```

---

## Tech Stack Additions

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Vector DB | Pinecone 3.0+ | Semantic search |
| Embeddings | OpenAI text-embedding-3-small | 1536-dim vectors |
| Database | SQLAlchemy 2.0+ | Metadata storage |
| Auth | Clerk + service token | Hosted auth + backend request signing |
| Doc Processing | PyPDF2, python-docx, unstructured | Load documents |
| Chunking | RecursiveCharacterTextSplitter | 1000 chars, 200 overlap |

---

## Database Schema (6 Tables)

### User Accounts
- `users` - Email, hashed password, active status

### Document Management
- `documents` - Metadata (filename, size, type, Pinecone namespace)
- `document_chunks` - Individual chunks with Pinecone IDs

### Chat History
- `chat_sessions` - Conversation grouping
- `chat_messages` - Q&A pairs with Pinecone IDs and Galileo traces

### RAG Evaluation
- `rag_evaluations` - Comprehensive RAG evaluation metrics (Phase 4)
  - Retrieval Quality: context_relevance, retrieval_precision, coverage
  - Hybrid Ranking: ranking_quality, source_diversity, confidence_calibration
  - Context Utilization: context_adherence, completeness, utilization scores

---

## Pinecone Structure

**Index**: `research-agent-rag`  
**Dimensions**: 1536  
**Metric**: Cosine  

**Namespaces**:
- `user_{user_id}_docs` - Document chunks
- `user_{user_id}_chat` - Chat history

**Vector Metadata**:
```python
# Documents
{
    "user_id": "uuid",
    "source_type": "document",
    "text": "chunk text",
    "document_name": "file.pdf",
    "chunk_index": 0
}

# Chat
{
    "user_id": "uuid",
    "source_type": "chat",
    "question": "...",
    "answer": "...",
    "timestamp": "..."
}
```

---

## File Structure

```
research-agent/
├── agent.py                    # Main workflow (TO BE MODIFIED)
├── api_server.py               # API server (TO BE MODIFIED)
│
├── backend/                    # NEW DIRECTORY ✅
│   ├── __init__.py             ✅
│   ├── models.py               ✅ Database models (RAG eval schema)
│   ├── database.py             ✅ Connection & sessions
│   ├── embeddings.py           ✅ OpenAI embeddings
│   ├── document_processor.py   ✅ Chunk & store
│   ├── documents.py            ✅ CRUD endpoints (Phase 2)
│   ├── chat_history.py         ✅ Chat endpoints + RAG eval storage (Phase 3)
│   └── auth.py                 ✅ Clerk service token guard (Phase 5)
│
├── rag_evaluators.py           ✅ RAG evaluation (Phase 4 COMPLETE)
├── requirements.txt            ✅ Updated with RAG deps
└── .env.example                ✅ Updated with Pinecone, DB, JWT
```

---

## Implementation Phases

### ✅ Phase 1: Core Infrastructure (COMPLETE)
- [x] Requirements.txt updated
- [x] Backend package created
- [x] Database models (6 tables)
- [x] Database connection (SQLite/PostgreSQL)
- [x] Embedding service
- [x] Document processor (chunking + Pinecone)
- [x] Environment variables

### 📋 Phase 2: Document RAG (NEXT)
- [x] Document upload endpoint
- [x] Modify AgentState for RAG fields
- [x] Add rag_retrieve_step()
- [x] Update curate_step() for hybrid ranking
- [x] Update API models

### 📋 Phase 3: Chat History RAG
- [x] Chat history endpoints
- [x] Store conversations in DB + Pinecone
- [x] Retrieve chat in rag_retrieve_step()
- [x] Include chat in hybrid ranking

### ✅ Phase 4: Galileo RAG Evaluation (COMPLETE)
- [x] Create rag_evaluators.py with 3 comprehensive evaluators
- [x] Evaluate retrieval quality (Context Relevance, Precision, Coverage)
- [x] Evaluate hybrid ranking (Quality, Diversity, Calibration)
- [x] Evaluate context utilization (Adherence, Completeness, Balance)
- [x] Integrate evaluators into workflow (3 steps)
- [x] Expand RAGEvaluation database schema (20+ metrics)
- [x] Implement database storage for all evaluations
- [x] Test end-to-end with all evaluation types

### ✅ Phase 5: Authentication
- [x] Clerk integration for sign-in/sign-up in Next.js
- [x] Service-to-service token protecting FastAPI endpoints
- [x] Automatic Clerk user provisioning in the database for namespace isolation
- [x] Frontend gating via SignedIn/SignedOut components and UserButton

---

## API Endpoints (Planned)

### Authentication
- Handled by Clerk via the Next.js frontend (`<ClerkProvider>`, `SignInButton`, `UserButton`).
- Next.js API routes call FastAPI with a shared `BACKEND_SERVICE_TOKEN` and Clerk user headers (`X-User-Id`, `X-User-Email`).
- FastAPI automatically provisions Clerk users into the local database for document/chat ownership.

### Documents
```
POST   /documents/upload      # Upload PDF/DOCX/TXT
GET    /documents             # List user's docs
DELETE /documents/{id}        # Delete doc
```

### Research (Enhanced)
```
POST /research
{
    "question": "...",
    "use_documents": true,        # NEW
    "document_ids": ["..."],      # NEW (optional)
    "use_chat_history": true,     # NEW
    "doc_retrieval_k": 5,         # NEW
    "chat_retrieval_k": 3         # NEW
}
```
> Requires `Authorization: Bearer <jwt>` header. Research agent derives `user_id` from the authenticated principal and prevents cross-tenant namespace access.

---

## Galileo Log Streams

### Existing
- `web-interface` - Main workflow
- `web-interface-eval` - Step evaluations

### Phase 4: RAG Evaluation Streams
- `rag-document-retrieval` - Retrieval quality evaluation (Context Relevance, Precision, Coverage)
- `rag-hybrid-ranking` - Hybrid ranking evaluation (Quality, Diversity, Calibration)
- `rag-context-utilization` - Context utilization evaluation (Adherence, Completeness, Balance)

---

## Environment Variables

```bash
# Core
GALILEO_API_KEY=...
OPENAI_API_KEY=...
TAVILY_API_KEY=...

# RAG
PINECONE_API_KEY=...

# Database
DATABASE_URL=sqlite:///./research_agent.db

# Auth
BACKEND_SERVICE_TOKEN=...  # Shared secret between Next.js API routes and FastAPI
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 3. Initialize database
python -m backend.database

# 4. Test components
python -m backend.embeddings
python -m backend.document_processor
```

---

## Design Decisions

1. **Single Pinecone index with namespaces** - Cost-effective vs per-user indexes
2. **Confidence boosting** - User docs +0.15, chat +0.30 vs web sources (increased in Phase 3)
3. **SQLite dev, PostgreSQL prod** - Easy local development
4. **Separate RAG evaluation streams** - Clean Galileo analytics (Phase 4)
5. **1000 char chunks, 200 overlap** - Balance context vs precision
6. **10 total sources** - Increased from 8 to allow more RAG sources in final answer

---

## Phase 4: RAG Evaluation Implementation

### Overview
Phase 4 adds comprehensive evaluation of all RAG operations using Galileo-recommended metrics. Three specialized evaluators track retrieval quality, ranking effectiveness, and context utilization.

### Evaluators (rag_evaluators.py)

#### 1. RAGEvaluator.evaluate_retrieval_quality()
**When**: After retrieving documents and chat history from Pinecone
**Evaluates**:
- Context Relevance (1-10): Do chunks have enough info to answer the query?
- Retrieval Precision (1-10): Are chunks actually relevant?
- Coverage (1-10): Do chunks cover all aspects of the query?

**Output**: `{context_relevance, retrieval_precision, coverage, overall_score, reasoning}`

#### 2. RAGEvaluator.evaluate_hybrid_ranking()
**When**: After merging and ranking web + document + chat sources
**Evaluates**:
- Ranking Quality (1-10): Are best sources at top?
- Source Diversity (1-10): Good mix of web/docs/chat?
- Confidence Calibration (1-10): Do scores match relevance?

**Output**: `{ranking_quality, source_diversity, confidence_calibration, source_distribution}`

#### 3. RAGEvaluator.evaluate_context_utilization()
**When**: After generating final answer
**Evaluates** (Galileo RAG metrics):
- Context Adherence (0-1): Answer sticks to context? (hallucination check)
- Completeness (0-1): All relevant context used? (recall metric)
- Document/Web Utilization (1-10): How well each source type used
- Source Balance (1-10): Good balance across source types

**Output**: `{context_adherence, completeness, document_utilization, web_utilization, balance, chunks_used}`

### Integration Points

1. **rag_retrieve_step()** (agent.py:455-495):
   - Calls `evaluate_retrieval_quality()`
   - Adds metrics to `rag_evaluation_metrics` state field

2. **curate_step()** (agent.py:771-817):
   - Calls `evaluate_hybrid_ranking()`
   - Adds metrics to `rag_evaluation_metrics` state field

3. **synthesize_step()** (agent.py:998-1049):
   - Calls `evaluate_context_utilization()`
   - Adds metrics to `rag_evaluation_metrics` state field

### Database Storage

**RAGEvaluation Table Schema**:
```python
class RAGEvaluation:
    # Identity
    id: str (UUID)
    message_id: str (FK to chat_messages)
    eval_type: str  # retrieval_quality | hybrid_ranking | context_utilization

    # Common
    overall_score: float
    latency: float
    reasoning: text
    galileo_trace_id: str

    # Retrieval Quality (1-10 scale)
    context_relevance: float
    retrieval_precision: float
    coverage: float
    num_doc_chunks: int
    num_chat_chunks: int

    # Hybrid Ranking (1-10 scale)
    ranking_quality: float
    source_diversity: float
    confidence_calibration: float
    num_sources: int
    source_distribution: json  # {web: N, document: N, chat: N}

    # Context Utilization (Galileo 0-1 scale)
    context_adherence: float  # No hallucinations = 1.0
    completeness: float  # All context used = 1.0
    document_utilization: float  # 1-10
    web_utilization: float  # 1-10
    source_balance: float  # 1-10
    chunks_used: json  # List of chunk IDs
    answer_length: int
```

**Storage Function** (`backend/chat_history.py:save_rag_evaluations()`):
- Called after `log_interaction()` in agent.run()
- Saves all evaluation metrics from `rag_evaluation_metrics` state
- Links evaluations to chat message via message_id

### Example Evaluation Flow

```
User asks: "What is machine learning?"
RAG enabled: use_chat_history=True

1. rag_retrieve_step()
   → Retrieves 3 chat chunks from Pinecone
   → evaluate_retrieval_quality()
     ✓ Context Relevance: 1/10 (chat about RAG not ML)
     ✓ Precision: 1/10
     ✓ Coverage: 1/10
   → Saves to rag_evaluation_metrics

2. curate_step()
   → Merges 8 web + 2 chat sources = 10 total
   → evaluate_hybrid_ranking()
     ✓ Ranking Quality: 8/10
     ✓ Source Diversity: 4/10 (mostly web)
     ✓ Confidence Calibration: 7/10
   → Saves to rag_evaluation_metrics

3. synthesize_step()
   → Generates answer using all 10 sources
   → evaluate_context_utilization()
     ✓ Context Adherence: 1.0 (no hallucinations!)
     ✓ Completeness: 0.80 (used 80% of context)
     ✓ Document Util: 5/10, Web Util: 5/10, Balance: 5/10
   → Saves to rag_evaluation_metrics

4. After agent.run() completes
   → log_interaction() returns (session_id, message_id)
   → save_rag_evaluations(message_id, rag_evaluation_metrics)
   → Database now has 3 RAGEvaluation records linked to chat message
```

### Verification

```bash
# Check evaluation counts by type
sqlite3 research_agent.db "
  SELECT eval_type, COUNT(*)
  FROM rag_evaluations
  GROUP BY eval_type;
"
# Output:
# retrieval_quality|8
# hybrid_ranking|4
# context_utilization|2

# Check Context Adherence scores (hallucination detection)
sqlite3 research_agent.db "
  SELECT context_adherence, completeness
  FROM rag_evaluations
  WHERE eval_type='context_utilization';
"
# Output:
# 1.0|1.0  (Perfect - no hallucinations, complete context usage!)
```

---

## Security

- ✅ Namespace isolation in Pinecone
- ✅ Database queries filtered by user_id
- ✅ Clerk session tokens enforced via middleware
- ✅ Shared backend service token for API calls
- ✅ FastAPI-Users authentication routers protecting documents/chat/research endpoints
- ✅ File type/size validation
- ✅ Environment variable secrets

---

**Next Step**: Phase 2 - Document RAG Integration

---

## Phase 5: Authentication Implementation

### Overview
Phase 5 now relies on Clerk for end-user authentication. The Next.js frontend renders Clerk-hosted sign-in/sign-up flows, while FastAPI trusts only requests signed with a shared `BACKEND_SERVICE_TOKEN` and annotated with Clerk user headers.

### Components

1. **Next.js Frontend**
   - `<ClerkProvider>` wraps the app
   - `SignedIn`/`SignedOut` fence the research UI and the auth call-to-action
   - `UserButton` handles account switching + sign-out without extra code

2. **API Routes**
   - Each `/api/*` handler calls `auth()`/`currentUser()` from `@clerk/nextjs/server`
   - Backend requests include `Authorization: Bearer ${BACKEND_SERVICE_TOKEN}` and `X-User-Id`/`X-User-Email`

3. **backend/auth.py**
   - Validates the shared token and headers
   - Auto-creates a local `User` record for new Clerk IDs and keeps emails in sync

4. **Security Hardening**
   - FastAPI rejects calls without the shared secret or missing Clerk headers
   - `.env.example` documents the new `BACKEND_SERVICE_TOKEN` requirement

### Usage Flow
1. Users sign in via Clerk (modal or dedicated page rendered by Next.js)
2. Frontend API routes detect the signed-in user via middleware and forward the request to FastAPI with the shared secret + Clerk identifiers
3. FastAPI provisions/loads the user, scopes DB queries by `user.id`, and preserves namespace isolation in Pinecone
