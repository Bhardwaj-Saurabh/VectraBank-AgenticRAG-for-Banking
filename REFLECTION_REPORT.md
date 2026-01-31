# Reflective Report: VectraBank Agentic RAG for Banking

## 1. Implementation Approach

### Architecture Overview
This project implements a multi-agent banking analysis system using Microsoft Semantic Kernel's `SequentialOrchestration` pattern. Six specialized AI agents process customer queries in sequence, each building on the previous agent's output to produce a comprehensive banking analysis report.

The system integrates three data sources:
- **Azure AI Foundry** (GPT-4.1) for agent reasoning and natural language generation
- **ChromaDB** as a local vector store for Retrieval Augmented Generation (RAG) over banking policy documents
- **Azure SQL Database** for structured customer and transaction data

### Design Decisions

**Sequential over Parallel Orchestration:** I chose sequential orchestration because banking analysis requires each specialist to build on prior findings. The Fraud Analyst needs the Data Gatherer's profile summary; the Synthesis Coordinator needs all five prior analyses. A parallel approach would lose this context chain.

**Graceful Fallback Pattern:** The `DataConnector` class degrades gracefully when Azure SQL is unavailable, falling back to hardcoded sample data. This ensures the system remains functional for development and testing without requiring a live database connection.

**Hybrid Search Strategy:** Rather than relying solely on semantic similarity, the `ChromaDBManager` implements hybrid search combining embedding-based semantic matching with keyword boosting. This improves retrieval accuracy for banking-specific terminology (e.g., "debt-to-income ratio") that pure semantic search might miss.

**Document Processing Pipeline:** The `rag_utils.py` module supports PDF, DOCX, and plain text formats through a unified `read_document_file()` interface. Documents are chunked with configurable overlap to preserve context at chunk boundaries before being embedded and stored in ChromaDB.

## 2. Evaluation Results: Strengths and Weaknesses

### Strengths

1. **Comprehensive Agent Coverage:** The six-agent architecture covers the full spectrum of banking analysis: data profiling, fraud detection, loan eligibility, customer support, enterprise risk, and executive synthesis. Each agent has detailed domain-specific instructions with structured output formats.

2. **Multi-Factor Risk Scoring:** The `_calculate_enhanced_risk_score()` method evaluates five independent risk dimensions (income, credit score, tenure, product diversification, transaction patterns) and produces a normalized 0-1 score, providing more nuanced assessment than single-factor approaches.

3. **RAG-Enhanced Context:** By retrieving relevant policy documents from ChromaDB and injecting them into agent prompts, the system grounds recommendations in actual banking policies rather than relying solely on the LLM's training data.

4. **Robust Testing:** The `--all` flag runs 29 component-level tests validating each subsystem independently, followed by full end-to-end scenario validation with automated pass/fail criteria.

### Weaknesses

1. **Agent Context Window Limitations:** In sequential orchestration, each agent receives the full accumulated context from prior agents. For complex queries, this can approach token limits, potentially causing truncation of earlier agent contributions.

2. **Static Customer Profiles:** When Azure SQL is unavailable, the system falls back to three hardcoded customer profiles. A production system would need a more robust caching layer or local database mirror.

3. **Single-Threaded Agent Execution:** The sequential pattern means total latency is the sum of all six agent calls (~30-45 seconds). Agents that don't depend on each other (e.g., Fraud Analyst and Loan Analyst) could theoretically run in parallel.

4. **Embedding Model Mismatch:** ChromaDB uses its default embedding model locally, while the system also configures Azure's `text-embedding-3-small`. Ideally the same model should be used for both storage and retrieval.

## 3. Suggestions for Improvement

### Suggestion 1: Implement Selective Agent Activation
Currently all six agents run for every query regardless of relevance. A query classifier could determine which agents to activate based on the query type. For example, a simple balance inquiry doesn't need fraud analysis or loan evaluation. This would reduce latency and API costs by adding a lightweight classification step before orchestration that maps query intent to a subset of agents.

### Suggestion 2: Add Persistent Conversation Memory
The current system treats each query independently with no memory of prior interactions. Adding a conversation history store (e.g., using CosmosDB or a local SQLite database) would allow agents to reference previous analyses for the same customer, enabling more personalized recommendations and trend detection over time. This could be implemented by extending `SharedState` to persist interaction history across sessions.

## 4. Azure Service and ChromaDB Integration Challenges

### Challenge 1: Azure SQL Connectivity
**Problem:** The `pyodbc` library requires the `unixodbc` system dependency on macOS, which is not installed by default, causing `ImportError` on initial setup.
**Solution:** Installed `unixodbc` via Homebrew (`brew install unixodbc`) and implemented a graceful fallback pattern so the system functions with sample data when SQL is unavailable.

### Challenge 2: ChromaDB Duplicate Document Ingestion
**Problem:** Re-running the system would attempt to re-add documents to ChromaDB collections, causing duplicate chunk errors.
**Solution:** Added a `load_enhanced_documents()` method that checks collection statistics before loading, skipping the ingestion step if documents are already present.

### Challenge 3: Agent Response Latency
**Problem:** Sequential orchestration with six agents calling Azure AI Foundry resulted in 30-45 second total processing time per query.
**Solution:** Accepted this as a trade-off for the sequential context-building pattern. Added processing time tracking in `processing_metrics` to monitor performance, and set a 180-second timeout as a safety limit.

## 5. Conclusion

This project demonstrates a functional multi-agent RAG system for banking analysis that integrates Azure cloud services with local vector search. The sequential orchestration pattern provides coherent, context-rich analyses at the cost of latency. The modular architecture (separate connector, manager, and utility modules) supports maintainability and testing. Key areas for future work include selective agent activation, persistent memory, and embedding model alignment.
