import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import uuid
from datetime import datetime

class ChromaDBManager:
    """Enhanced ChromaDB manager with banking-specific functionality"""
    
    def __init__(self, persist_directory: str = "./chroma_db_banking"):
        self.persist_directory = persist_directory
        self.client = None
        self.collections = {}
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client with proper settings"""
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            print("✅ ChromaDB client initialized successfully")
            self._initialize_collections()
        except Exception as e:
            print(f"❌ ChromaDB initialization failed: {e}")
            self.client = None
    
    def _initialize_collections(self):
        """Initialize banking-specific collections"""
        banking_collections = {
            "fraud_detection": "Fraud detection policies and patterns",
            "loan_policies": "Loan eligibility and credit policies",
            "customer_support": "Customer service and support guidelines", 
            "risk_assessment": "Risk analysis and compliance policies",
            "transaction_monitoring": "Transaction patterns and monitoring rules",
            "compliance": "Regulatory compliance requirements"
        }
        
        for name, description in banking_collections.items():
            try:
                collection = self.client.get_or_create_collection(
                    name=name,
                    metadata={"description": description, "type": "banking"}
                )
                self.collections[name] = collection
                print(f"  ✅ Collection '{name}' initialized")
            except Exception as e:
                print(f"  ❌ Failed to initialize collection '{name}': {e}")
    
    def determine_collection(self, filename: str, content: str) -> str:
        """Determine the appropriate collection based on content analysis"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["fraud", "suspicious", "security", "unauthorized"]):
            return "fraud_detection"
        elif any(word in content_lower for word in ["loan", "credit", "eligibility", "borrow"]):
            return "loan_policies"
        elif any(word in content_lower for word in ["support", "customer", "service", "help"]):
            return "customer_support"
        elif any(word in content_lower for word in ["risk", "compliance", "regulation", "mitigation"]):
            return "risk_assessment"
        elif any(word in content_lower for word in ["transaction", "monitoring", "pattern", "alert"]):
            return "transaction_monitoring"
        else:
            return "compliance"
    
    async def chunk_and_store_document(self, filename: str, content: str, collection_type: str) -> int:
        """Chunk document and store in appropriate collection"""
        if collection_type not in self.collections:
            await self.create_collection(collection_type)
        
        # Simple chunking strategy for banking documents
        chunks = self._chunk_document(content, filename)
        
        if not chunks:
            return 0
        
        # Prepare documents for storage
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename}_{i}_{uuid.uuid4().hex[:8]}"
            documents.append(chunk)
            metadatas.append({
                "filename": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "collection": collection_type,
                "timestamp": datetime.now().isoformat()
            })
            ids.append(chunk_id)
        
        # Store in ChromaDB
        try:
            self.collections[collection_type].add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"  ✅ Stored {len(documents)} chunks from {filename} in {collection_type}")
            return len(documents)
        except Exception as e:
            print(f"  ❌ Failed to store chunks from {filename}: {e}")
            return 0
    
    def _chunk_document(self, content: str, filename: str) -> List[str]:
        """Split document into chunks for banking content"""
        # Simple paragraph-based chunking for policy documents
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        chunks = []
        
        current_chunk = ""
        for paragraph in paragraphs:
            # If adding this paragraph would make the chunk too long, save current chunk
            if len(current_chunk) + len(paragraph) > 1000 and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            current_chunk += paragraph + "\n\n"
        
        # Add the last chunk if it exists
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # If no paragraphs were found, split by sentences
        if not chunks:
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) > 800 and current_chunk:
                    chunks.append(current_chunk.strip() + ".")
                    current_chunk = ""
                current_chunk += sentence + ". "
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
        
        return chunks
    
    async def semantic_search(self, query: str, collection_names: List[str], top_k: int = 3) -> List[Dict]:
        """Enhanced semantic search with banking context"""
        all_results = []
        
        for collection_name in collection_names:
            if collection_name not in self.collections:
                continue
            
            try:
                search_results = self.collections[collection_name].query(
                    query_texts=[query],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )
                
                if search_results and search_results['documents']:
                    for i, (document, metadata, distance) in enumerate(zip(
                        search_results['documents'][0],
                        search_results['metadatas'][0],
                        search_results['distances'][0] if search_results['distances'] else [0.1] * len(search_results['documents'][0])
                    )):
                        all_results.append({
                            "document": document,
                            "metadata": metadata,
                            "distance": distance,
                            "collection": collection_name,
                            "relevance_score": 1 - distance,
                            "filename": metadata.get('filename', 'Unknown'),
                            "chunk_info": f"Chunk {metadata.get('chunk_index', 0)} of {metadata.get('total_chunks', 1)}"
                        })
            except Exception as e:
                print(f"  ❌ Search error in {collection_name}: {e}")
        
        # Sort by relevance (higher score is better)
        all_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return all_results[:top_k * len(collection_names)]
    
    async def hybrid_search(self, query: str, collection_names: List[str], top_k: int = 3) -> List[Dict]:
        """Hybrid search combining semantic and keyword matching"""
        semantic_results = await self.semantic_search(query, collection_names, top_k)
        
        # Add keyword-based relevance boost
        query_keywords = set(query.lower().split())
        for result in semantic_results:
            document_text = result["document"].lower()
            keyword_matches = len([kw for kw in query_keywords if kw in document_text])
            result["keyword_boost"] = keyword_matches * 0.1
            result["final_score"] = result["relevance_score"] + result["keyword_boost"]
        
        # Re-sort by final score
        semantic_results.sort(key=lambda x: x["final_score"], reverse=True)
        return semantic_results
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get detailed statistics for all collections"""
        stats = {}
        for name, collection in self.collections.items():
            try:
                count = collection.count()
                stats[name] = {
                    "document_count": count,
                    "type": "chromadb",
                    "status": "active" if count > 0 else "empty"
                }
            except Exception as e:
                stats[name] = {
                    "document_count": 0,
                    "type": "error",
                    "status": f"error: {str(e)}"
                }
        return stats
    
    async def create_collection(self, name: str, description: str = ""):
        """Create a new collection"""
        try:
            collection = self.client.get_or_create_collection(
                name=name,
                metadata={"description": description, "type": "banking", "created": datetime.now().isoformat()}
            )
            self.collections[name] = collection
            return collection
        except Exception as e:
            print(f"❌ Error creating collection {name}: {e}")
            return None
    
    async def delete_collection(self, name: str):
        """Delete a collection"""
        try:
            if name in self.collections:
                self.client.delete_collection(name)
                del self.collections[name]
                print(f"✅ Deleted collection: {name}")
        except Exception as e:
            print(f"❌ Error deleting collection {name}: {e}")
    
    async def get_document_chunks(self, filename: str, collection_name: str) -> List[Dict]:
        """Get all chunks for a specific document"""
        try:
            results = self.collections[collection_name].get(
                where={"filename": filename},
                include=["documents", "metadatas"]
            )
            
            chunks = []
            for doc, metadata in zip(results['documents'], results['metadatas']):
                chunks.append({
                    "content": doc,
                    "metadata": metadata
                })
            
            return sorted(chunks, key=lambda x: x["metadata"].get("chunk_index", 0))
        except Exception as e:
            print(f"❌ Error getting chunks for {filename}: {e}")
            return []
