# SafeBuild: Agentic AI Safety Inspection System (Azure)

## Overview
SafeBuild is an enterprise-style agentic AI system built on Azure that combines:
- AI-powered safety Q&A (RAG)
- Intelligent inspection intake (structured data capture)
- Orchestrated agent routing
- Real-time Azure SQL data persistence

This project demonstrates production-ready patterns for building AI + data engineering systems using Azure services.

---

## Architecture

User  
→ Azure Container App (FastAPI App)  
→ Orchestrator (intent routing)  
→  
1) Safety Agent (RAG via Azure AI Search + LLM)  
2) Intake Agent (structured extraction → REST → DAB)  
→  
DAB (Data API Builder)  
→ Azure SQL Database (bronze.SafetyInspections)

---

## Key Features

### 1. Safety Q&A Agent
- Uses Azure AI Search for retrieval (RAG)
- Grounded responses based on indexed safety content
- Handles construction safety queries

### 2. Inspection Intake Agent
- Converts natural language into structured inspection records
- Validates required fields
- Writes directly to Azure SQL via DAB REST API

### 3. Smart Assistant (Orchestrator)
- Routes user intent dynamically:
  - Q&A vs Inspection creation
- Maintains in-session conversation context

### 4. Web UI
- Built with FastAPI + Jinja
- Tabs for:
  - Smart Assistant
  - Safety Q&A
  - Inspection Intake

---

## Tech Stack

- Azure Container Apps
- Azure OpenAI-compatible endpoint (Grok model)
- Azure AI Search (RAG)
- Azure SQL Database
- Data API Builder (DAB)
- FastAPI + Jinja2
- Docker + Azure Container Registry

---

## Data Flow

1. User submits query or inspection input
2. Orchestrator detects intent
3. Routes to:
   - RAG pipeline (Q&A), or
   - Extraction → REST API (inspection)
4. Data stored in Azure SQL
5. Response returned to UI

---

## Evaluation & Monitoring (Concept)

- Operational metrics via Foundry (latency, tokens, success rate)
- Manual evaluation baseline for:
  - Groundedness
  - Relevance
  - Coherence
- Drift tracking via repeated evaluation sets

---

## Project Status

- Phase 1: Core App — Complete  
- Phase 2: RAG — Working (keyword-based)  
- Phase 3: Agents + SQL + Orchestrator — Complete  

---

## Future Enhancements

- Vector search for improved RAG
- Persistent chat memory
- Automated evaluation pipelines
- Data drift monitoring (Azure ML)
- Multi-agent orchestration scaling

---

## How to Run (Local)

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```
## Future Enhancements

- Dockerized and deployed to Azure Container Apps
- Uses environment variables for secrets/config

## Author

Amanuel Birri |
Azure | AI Systems | Data Engineering