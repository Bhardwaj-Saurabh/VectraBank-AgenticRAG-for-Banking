from typing import Any, List, Dict
import re

def extract_banking_policies(docs: List[Dict]) -> Dict[str, Any]:
    """
    Extract and structure banking policies from loaded documents for Semantic Kernel
    """
    policies = {
        "loan_policies": {},
        "fraud_policies": {},
        "support_policies": {}
    }
    
    for doc in docs:
        if "Loan" in doc["id"]:
            policies["loan_policies"] = _parse_loan_policies(doc["text"])
        elif "Fraud" in doc["id"]:
            policies["fraud_policies"] = _parse_fraud_policies(doc["text"])
        elif "Support" in doc["id"]:
            policies["support_policies"] = _parse_support_policies(doc["text"])
    
    return policies

def _parse_loan_policies(text: str) -> Dict[str, Any]:
    """Parse loan eligibility policies from document text"""
    policies = {
        "income_requirements": {},
        "credit_score_ranges": {},
        "debt_to_income_limits": {},
        "document_requirements": [],
        "risk_tiers": {}
    }
    
    # Simple parsing logic - can be enhanced with NLP
    lines = text.split('\n')
    for line in lines:
        line_lower = line.lower()
        if "income" in line_lower and "minimum" in line_lower:
            # Extract numbers
            numbers = re.findall(r'\$?(\d+(?:,\d+)*(?:\.\d+)?)', line)
            if numbers:
                policies["income_requirements"]["minimum"] = float(numbers[0].replace(',', ''))
        elif "credit score" in line_lower:
            numbers = re.findall(r'\d+', line)
            if len(numbers) >= 2:
                policies["credit_score_ranges"] = {
                    "excellent": int(numbers[0]),
                    "good": int(numbers[1]) if len(numbers) > 1 else 700,
                    "fair": int(numbers[2]) if len(numbers) > 2 else 650
                }
        elif "debt" in line_lower and "income" in line_lower:
            numbers = re.findall(r'(\d+(?:\.\d+)?)%', line)
            if numbers:
                policies["debt_to_income_limits"]["maximum"] = float(numbers[0]) / 100
    
    return policies

def _parse_fraud_policies(text: str) -> Dict[str, Any]:
    """Parse fraud detection policies from document text"""
    policies = {
        "suspicious_patterns": [],
        "risk_thresholds": {},
        "escalation_procedures": [],
        "monitoring_rules": {}
    }
    
    lines = text.split('\n')
    for line in lines:
        line_lower = line.lower()
        if any(word in line_lower for word in ["large transaction", "unusual activity"]):
            policies["suspicious_patterns"].append("large_transactions")
        elif any(word in line_lower for word in ["multiple locations", "geographic"]):
            policies["suspicious_patterns"].append("unusual_locations")
        elif "threshold" in line_lower:
            numbers = re.findall(r'\$?(\d+(?:,\d+)*(?:\.\d+)?)', line)
            if numbers:
                policies["risk_thresholds"]["transaction_amount"] = float(numbers[0].replace(',', ''))
    
    return policies

def _parse_support_policies(text: str) -> Dict[str, Any]:
    """Parse customer support policies from document text"""
    policies = {
        "response_times": {},
        "escalation_paths": [],
        "self_service_options": [],
        "contact_methods": []
    }
    
    lines = text.split('\n')
    for line in lines:
        line_lower = line.lower()
        if "response" in line_lower and "time" in line_lower:
            if "hour" in line_lower:
                policies["response_times"]["standard"] = "24 hours"
            if "minute" in line_lower:
                policies["response_times"]["urgent"] = "30 minutes"
        elif any(word in line_lower for word in ["escalate", "supervisor"]):
            policies["escalation_paths"].append("supervisor_escalation")
        elif any(word in line_lower for word in ["self-service", "automated"]):
            policies["self_service_options"].append("online_portal")
    
    return policies

def create_semantic_kernel_context(policies: Dict[str, Any]) -> str:
    """
    Create a context string from policies for Semantic Kernel prompts
    """
    context_parts = []
    
    # Loan policies context
    if policies["loan_policies"]:
        loan_ctx = "Loan Policies:\n"
        if policies["loan_policies"].get("income_requirements"):
            min_income = policies["loan_policies"]["income_requirements"].get("minimum")
            if min_income:
                loan_ctx += f"- Minimum income: ${min_income:,.2f}\n"
        if policies["loan_policies"].get("credit_score_ranges"):
            scores = policies["loan_policies"]["credit_score_ranges"]
            loan_ctx += f"- Credit score ranges: Excellent ({scores.get('excellent', 750)}+), Good ({scores.get('good', 700)}+)\n"
        context_parts.append(loan_ctx)
    
    # Fraud policies context
    if policies["fraud_policies"]:
        fraud_ctx = "Fraud Detection Policies:\n"
        if policies["fraud_policies"].get("suspicious_patterns"):
            patterns = policies["fraud_policies"]["suspicious_patterns"]
            fraud_ctx += f"- Monitor for: {', '.join(patterns)}\n"
        if policies["fraud_policies"].get("risk_thresholds"):
            threshold = policies["fraud_policies"]["risk_thresholds"].get("transaction_amount")
            if threshold:
                fraud_ctx += f"- High risk threshold: ${threshold:,.2f}\n"
        context_parts.append(fraud_ctx)
    
    # Support policies context
    if policies["support_policies"]:
        support_ctx = "Customer Support Policies:\n"
        if policies["support_policies"].get("response_times"):
            times = policies["support_policies"]["response_times"]
            support_ctx += f"- Response times: Standard ({times.get('standard', '24 hours')}), Urgent ({times.get('urgent', '30 minutes')})\n"
        if policies["support_policies"].get("self_service_options"):
            options = policies["support_policies"]["self_service_options"]
            support_ctx += f"- Self-service: {', '.join(options)}\n"
        context_parts.append(support_ctx)
    
    return "\n".join(context_parts) if context_parts else "No specific policies loaded."

def validate_against_policies(customer_data: Dict[str, Any], policies: Dict[str, Any], query_type: str) -> Dict[str, Any]:
    """
    Validate customer data against banking policies
    """
    validation_result = {
        "compliant": True,
        "violations": [],
        "warnings": [],
        "recommendations": []
    }
    
    if query_type == "loan":
        loan_policies = policies.get("loan_policies", {})
        income = customer_data.get("income", 0)
        transactions = customer_data.get("transactions", [])
        
        # Check income requirements
        min_income = loan_policies.get("income_requirements", {}).get("minimum", 0)
        if income < min_income:
            validation_result["compliant"] = False
            validation_result["violations"].append(f"Income ${income:,.2f} below minimum requirement ${min_income:,.2f}")
        
        # Check debt-to-income
        monthly_debits = sum(tx.get('amount', 0) for tx in transactions if tx.get('type') == 'debit')
        monthly_income = income / 12
        dti = monthly_debits / monthly_income if monthly_income > 0 else 0
        max_dti = loan_policies.get("debt_to_income_limits", {}).get("maximum", 0.5)
        
        if dti > max_dti:
            validation_result["warnings"].append(f"Debt-to-income ratio {dti:.2%} exceeds recommended maximum {max_dti:.2%}")
    
    elif query_type == "fraud":
        fraud_policies = policies.get("fraud_policies", {})
        transactions = customer_data.get("transactions", [])
        threshold = fraud_policies.get("risk_thresholds", {}).get("transaction_amount", 1000)
        
        large_transactions = [tx for tx in transactions if tx.get('amount', 0) > threshold]
        if large_transactions:
            validation_result["warnings"].append(f"Found {len(large_transactions)} transactions exceeding ${threshold:,.2f} threshold")
    
    return validation_result

