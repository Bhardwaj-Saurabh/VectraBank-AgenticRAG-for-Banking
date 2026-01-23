from typing import Dict, Any, List
from datetime import datetime
import threading

class SharedState:
    """
    Thread-safe shared state for banking agents
    """
    
    def __init__(self):
        self._lock = threading.RLock()
        self.interactions: Dict[str, List[Dict[str, Any]]] = {}
        self.customer_data: Dict[str, Dict[str, Any]] = {}
        self.system_metrics: Dict[str, Any] = {
            "total_interactions": 0,
            "successful_processing": 0,
            "failed_processing": 0,
            "start_time": datetime.now().isoformat()
        }
    
    def update_interaction(self, customer_id: str, interaction_data: Dict[str, Any]):
        """Add or update customer interaction"""
        with self._lock:
            if customer_id not in self.interactions:
                self.interactions[customer_id] = []
            
            self.interactions[customer_id].append(interaction_data)
            self.system_metrics["total_interactions"] += 1
            self.system_metrics["successful_processing"] += 1
    
    def record_failure(self, customer_id: str, error: str):
        """Record processing failure"""
        with self._lock:
            self.system_metrics["failed_processing"] += 1
            
            failure_record = {
                "customer_id": customer_id,
                "error": error,
                "timestamp": datetime.now().isoformat(),
                "type": "processing_failure"
            }
            
            if customer_id not in self.interactions:
                self.interactions[customer_id] = []
            
            self.interactions[customer_id].append(failure_record)
    
    def get_customer_interactions(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get all interactions for a customer"""
        with self._lock:
            return self.interactions.get(customer_id, [])
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        with self._lock:
            return self.system_metrics.copy()
    
    def update_customer_data(self, customer_id: str, data: Dict[str, Any]):
        """Update customer data cache"""
        with self._lock:
            self.customer_data[customer_id] = data
    
    def get_customer_data(self, customer_id: str) -> Dict[str, Any]:
        """Get cached customer data"""
        with self._lock:
            return self.customer_data.get(customer_id, {})