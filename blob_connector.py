import os
import json
from typing import List, Dict, Optional
from datetime import datetime

class BlobStorageConnector:
    """Enhanced blob storage connector with banking document management"""
    
    def __init__(self, storage_path: str = "./banking_documents"):
        self.storage_path = storage_path
        self._ensure_storage_path()
        self.documents = self._load_document_registry()
    
    def _ensure_storage_path(self):
        """Ensure storage directory exists"""
        os.makedirs(self.storage_path, exist_ok=True)
    
    def _load_document_registry(self) -> Dict:
        """Load or create document registry"""
        registry_path = os.path.join(self.storage_path, "document_registry.json")
        if os.path.exists(registry_path):
            try:
                with open(registry_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_document_registry(self):
        """Save document registry"""
        registry_path = os.path.join(self.storage_path, "document_registry.json")
        with open(registry_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, indent=2)
    
    def upload_sample_documents(self):
        """Upload comprehensive banking sample documents"""
        sample_documents = {
            "fraud_detection_policy_v2.md": {
                "content": """
# Fraud Detection Policy v2.0
## Comprehensive Fraud Prevention Guidelines

### Risk Categories:
- **Low Risk**: Transactions under $500, familiar locations
- **Medium Risk**: Transactions $500-$2,000, new locations  
- **High Risk**: Transactions over $2,000, international, unusual patterns

### Monitoring Protocols:
1. Real-time transaction monitoring
2. Behavioral pattern analysis
3. Geographic anomaly detection
4. Device fingerprinting
5. Biometric verification for high-value transactions

### Escalation Matrix:
- Level 1: Automated flagging and customer notification
- Level 2: Manual review by fraud team
- Level 3: Account freeze and security investigation
- Level 4: Law enforcement coordination

### Response Timelines:
- High risk: Immediate action required
- Medium risk: 2-hour response window
- Low risk: 24-hour review cycle
                """,
                "metadata": {
                    "type": "fraud",
                    "version": "2.0",
                    "effective_date": "2024-01-01",
                    "department": "Security"
                }
            },
            
            "loan_eligibility_framework.md": {
                "content": """
# Loan Eligibility Framework
## Comprehensive Credit Assessment

### Income Requirements:
| Tier | Minimum Income | Debt-to-Income | Employment History |
|------|----------------|----------------|-------------------|
| A+   | $100,000+      | <30%           | 3+ years          |
| A    | $75,000+       | <35%           | 2+ years          |
| B    | $50,000+       | <40%           | 1+ years          |
| C    | $30,000+       | <45%           | 6+ months         |

### Credit Score Tiers:
- **Excellent (750+)**: 3.5% APR, 90% LTV
- **Good (700-749)**: 4.5% APR, 85% LTV
- **Fair (650-699)**: 6.0% APR, 80% LTV
- **Review (<650)**: Case-by-case assessment

### Documentation Matrix:
- **Basic**: Government ID, Income verification
- **Standard**: + Bank statements (3 months), Employment verification
- **Comprehensive**: + Tax returns (2 years), Asset documentation
- **Premium**: + Business financials, Investment portfolios
                """,
                "metadata": {
                    "type": "loans", 
                    "version": "1.2",
                    "effective_date": "2024-01-15",
                    "department": "Lending"
                }
            },
            
            "customer_support_protocols.md": {
                "content": """
# Customer Support Protocols
## Service Excellence Framework

### Priority Classification:
- **P0 - Critical**: Account security, Fraud, System outages
- **P1 - High**: Account access, Transaction issues, Loan applications
- **P2 - Medium**: General inquiries, Service changes, Documentation
- **P3 - Low**: Information requests, Feedback, Educational

### Response Time Standards:
- P0: 15 minutes (24/7 coverage)
- P1: 2 hours (business hours)
- P2: 24 hours (business days)
- P3: 48 hours (standard processing)

### Escalation Path:
1. **Tier 1**: Automated systems & basic support
2. **Tier 2**: Specialized banking agents
3. **Tier 3**: Senior banking specialists
4. **Tier 4**: Department management
5. **Tier 5**: Executive review

### Quality Standards:
- First contact resolution: 85% target
- Customer satisfaction: 90% target
- Resolution time: Within SLA 95% of cases
                """,
                "metadata": {
                    "type": "support",
                    "version": "1.1", 
                    "effective_date": "2024-02-01",
                    "department": "Customer Service"
                }
            },
            
            "risk_assessment_framework.md": {
                "content": """
# Risk Assessment Framework
## Comprehensive Risk Management

### Risk Categories:
1. **Credit Risk**: Borrower default probability
2. **Market Risk**: Economic and market fluctuations  
3. **Operational Risk**: Internal process failures
4. **Compliance Risk**: Regulatory violations
5. **Reputational Risk**: Brand and trust impact

### Scoring Matrix:
| Risk Level | Probability | Impact | Mitigation Required |
|------------|-------------|--------|-------------------|
| Low        | <10%        | Minor  | Basic monitoring   |
| Medium     | 10-30%      | Moderate | Enhanced controls |
| High       | 30-60%      | Major  | Active management  |
| Critical   | >60%        | Severe | Immediate action   |

### Assessment Frequency:
- High-risk customers: Quarterly reviews
- Medium-risk: Semi-annual reviews  
- Low-risk: Annual reviews
- New customers: Initial 90-day intensive monitoring

### Mitigation Strategies:
- Diversification of portfolio
- Enhanced due diligence
- Insurance and hedging
- Compliance automation
- Continuous monitoring
                """,
                "metadata": {
                    "type": "risk",
                    "version": "2.1",
                    "effective_date": "2024-01-20", 
                    "department": "Risk Management"
                }
            },
            
            "transaction_monitoring_guide.md": {
                "content": """
# Transaction Monitoring Guide
## AML & Fraud Prevention

### Monitoring Triggers:
- **Amount-based**: Transactions > $10,000
- **Frequency-based**: >10 transactions/hour
- **Pattern-based**: Unusual time/location patterns
- **Relationship-based**: New payees/beneficiaries

### Alert Categories:
1. **High Priority**: Structuring, Rapid movement, Known fraud patterns
2. **Medium Priority**: Geographic anomalies, New relationships
3. **Low Priority**: Minor deviations, One-time occurrences

### Investigation Protocols:
- **Phase 1**: Automated pattern recognition
- **Phase 2**: Manual review and customer contact
- **Phase 3**: Enhanced due diligence
- **Phase 4**: Regulatory reporting (if required)

### Reporting Requirements:
- SAR (Suspicious Activity Report): Within 30 days
- CTR (Currency Transaction Report): >$10,000 transactions
- International transfers: Enhanced monitoring
                """,
                "metadata": {
                    "type": "transaction_monitoring",
                    "version": "1.3",
                    "effective_date": "2024-03-01",
                    "department": "Compliance"
                }
            }
        }
        
        for filename, doc_info in sample_documents.items():
            file_path = os.path.join(self.storage_path, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(doc_info["content"])
            
            # Update registry
            self.documents[filename] = {
                **doc_info["metadata"],
                "upload_date": datetime.now().isoformat(),
                "file_size": len(doc_info["content"]),
                "status": "active"
            }
        
        self._save_document_registry()
        print(f"✅ Uploaded {len(sample_documents)} sample banking documents")
    
    def list_documents(self) -> List[str]:
        """List all available documents"""
        return list(self.documents.keys())
    
    def get_document_content(self, doc_name: str) -> Optional[str]:
        """Get document content with error handling"""
        if doc_name not in self.documents:
            return None
        
        file_path = os.path.join(self.storage_path, doc_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"❌ Error reading document {doc_name}: {e}")
            return None
    
    def get_document_metadata(self, doc_name: str) -> Optional[Dict]:
        """Get document metadata"""
        return self.documents.get(doc_name)
    
    def upload_custom_document(self, filename: str, content: str, metadata: Dict = None):
        """Upload a custom banking document"""
        file_path = os.path.join(self.storage_path, filename)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Update registry
            self.documents[filename] = {
                **(metadata or {}),
                "upload_date": datetime.now().isoformat(),
                "file_size": len(content),
                "status": "active"
            }
            
            self._save_document_registry()
            print(f"✅ Uploaded custom document: {filename}")
            return True
        except Exception as e:
            print(f"❌ Error uploading document {filename}: {e}")
            return False
    
    def delete_document(self, doc_name: str) -> bool:
        """Delete a document"""
        if doc_name not in self.documents:
            return False
        
        file_path = os.path.join(self.storage_path, doc_name)
        try:
            os.remove(file_path)
            del self.documents[doc_name]
            self._save_document_registry()
            print(f"✅ Deleted document: {doc_name}")
            return True
        except Exception as e:
            print(f"❌ Error deleting document {doc_name}: {e}")
            return False
    
    def search_documents(self, query: str) -> List[Dict]:
        """Search documents by content and metadata"""
        results = []
        query_lower = query.lower()
        
        for doc_name, metadata in self.documents.items():
            content = self.get_document_content(doc_name)
            if not content:
                continue
            
            # Simple keyword matching
            content_matches = query_lower in content.lower()
            metadata_matches = any(
                query_lower in str(value).lower() 
                for value in metadata.values() 
                if isinstance(value, str)
            )
            
            if content_matches or metadata_matches:
                results.append({
                    "filename": doc_name,
                    "metadata": metadata,
                    "content_preview": content[:200] + "..." if len(content) > 200 else content,
                    "match_type": "content" if content_matches else "metadata"
                })
        
        return results
    
    def get_document_stats(self) -> Dict:
        """Get document statistics"""
        stats = {
            "total_documents": len(self.documents),
            "documents_by_type": {},
            "documents_by_department": {},
            "total_size_bytes": 0
        }
        
        for doc_name, metadata in self.documents.items():
            doc_type = metadata.get('type', 'unknown')
            department = metadata.get('department', 'unknown')
            file_size = metadata.get('file_size', 0)
            
            stats["documents_by_type"][doc_type] = stats["documents_by_type"].get(doc_type, 0) + 1
            stats["documents_by_department"][department] = stats["documents_by_department"].get(department, 0) + 1
            stats["total_size_bytes"] += file_size
        
        return stats