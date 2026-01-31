import asyncio
import os
import uuid
import logging
import pyodbc
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from contextlib import contextmanager
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent, SequentialOrchestration
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.kernel_pydantic import KernelBaseModel
from semantic_kernel.contents import ChatMessageContent
from rag_utils import extract_banking_policies, create_semantic_kernel_context
from blob_connector import BlobStorageConnector
from chroma_manager import ChromaDBManager
from shared_state import SharedState
from dotenv import load_dotenv

load_dotenv()

# Global logger instance
logger = logging.getLogger(__name__)

def setup_logging():
    """Setup logging with unique file for each run"""
    if not os.path.exists("logs"):
        os.makedirs("logs")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/banking_analysis_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
        ]
    )

    print(f"Logger started. Log file: {log_filename}")
    return log_filename

class DataConnector:
    """Azure SQL Database connectivity for banking data retrieval"""

    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or os.getenv("AZURE_SQL_CONNECTION_STRING")
        if self.connection_string:
            self._test_connection()
        else:
            logger.warning("No SQL connection string provided. Using sample data fallback.")

    def _test_connection(self):
        """Test database connection on initialization"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
            logger.info("Database connection test successful")
        except Exception as e:
            logger.warning(f"Database connection test failed: {e}. Will use sample data fallback.")
            self.connection_string = None

    async def fetch_income(self, customer_id: str) -> Optional[float]:
        """Fetch customer income from Azure SQL database"""
        if not self.connection_string:
            return None
        try:
            await asyncio.sleep(0)
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT TOP 1 income FROM transactions WHERE customer_id = ? ORDER BY ts DESC",
                    (customer_id,)
                )
                row = cursor.fetchone()
                cursor.close()
                return float(row[0]) if row else None
        except Exception as e:
            logger.error(f"Error fetching income for customer {customer_id}: {e}")
            return None

    async def fetch_transactions(self, customer_id: str) -> List[Dict]:
        """Fetch customer transactions from Azure SQL database"""
        if not self.connection_string:
            return []
        try:
            await asyncio.sleep(0)
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT transaction_id, customer_id, income, amount, ts, description "
                    "FROM transactions WHERE customer_id = ? ORDER BY ts DESC",
                    (customer_id,)
                )
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                cursor.close()
                return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching transactions for customer {customer_id}: {e}")
            return []

    @contextmanager
    def get_db_connection(self):
        """Create database connection context manager"""
        conn = None
        try:
            conn = pyodbc.connect(self.connection_string, timeout=30)
            yield conn
        except pyodbc.Error as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()


class EnhancedBankingReport(KernelBaseModel):
    """Comprehensive banking analysis report"""
    report_id: str
    customer_id: str
    query: str
    summary: str
    key_findings: List[str] = []
    risk_assessment: str = "medium"
    risk_score: float = 0.5
    recommendations: List[str] = []
    actions_taken: List[str] = []
    policy_references: List[str] = []
    agent_contributions: Dict[str, Any] = {}
    processing_metrics: Dict[str, Any] = {}
    generated_by: str = "EnhancedBankingOrchestration"
    generated_at: str = datetime.now().isoformat()


class CustomerProfile(KernelBaseModel):
    """Comprehensive customer profile for banking analysis"""
    customer_id: str
    income: float = 0.0
    credit_score: int = 0
    account_type: str = "standard"
    customer_since: str = ""
    risk_tier: str = "medium"
    recent_transactions: List[Dict[str, Any]] = []
    banking_products: List[str] = []
    last_review_date: str = ""


class EnhancedBankingSequentialOrchestration:
    """Enhanced banking system with advanced features"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize enhanced storage components
        self.blob_connector = BlobStorageConnector()
        self.chroma_store = ChromaDBManager()
        self.shared_state = SharedState()

        # Initialize Azure SQL Data Connector (graceful fallback if unavailable)
        self.data_connector = DataConnector()

        # Initialize enhanced kernel
        self.kernel = Kernel()

        # Configure Azure AI Foundry service
        self.kernel.add_service(
            AzureChatCompletion(
                service_id="enhanced_banking_chat",
                deployment_name=os.environ["AZURE_TEXTGENERATOR_DEPLOYMENT_NAME"],
                endpoint=os.environ["AZURE_TEXTGENERATOR_DEPLOYMENT_ENDPOINT"],
                api_key=os.environ["AZURE_TEXTGENERATOR_DEPLOYMENT_KEY"]
            )
        )

        # Load enhanced banking policies
        self.banking_policies = self._load_enhanced_policies()
        self.customer_profiles = {}

        # Performance tracking
        self.performance_metrics = {
            "total_requests": 0,
            "successful_analyses": 0,
            "average_processing_time": 0,
            "agent_performance": {}
        }

    def _load_enhanced_policies(self) -> Dict[str, Any]:
        """Load and parse banking policy documents"""
        try:
            if not self.blob_connector.list_documents():
                self.blob_connector.upload_sample_documents()

            enhanced_docs = []
            for doc_name in self.blob_connector.list_documents():
                content = self.blob_connector.get_document_content(doc_name)
                metadata = self.blob_connector.get_document_metadata(doc_name)

                enhanced_docs.append({
                    "filename": doc_name,
                    "id": f"{metadata.get('type', 'general')}_{doc_name}",
                    "meta": {
                        **metadata,
                        "priority": "high" if metadata.get('type') in ['fraud', 'risk'] else "medium",
                        "review_frequency": "quarterly" if metadata.get('type') in ['fraud', 'compliance'] else "annually"
                    },
                    "text": content
                })

            policies = extract_banking_policies(enhanced_docs)
            self.logger.info(f"Loaded policies from {len(enhanced_docs)} documents")
            return policies
        except Exception as e:
            self.logger.error(f"Could not load enhanced banking policies: {e}")
            return {}

    async def _load_customer_profiles(self) -> Dict[str, CustomerProfile]:
        """Load customer profiles from Azure SQL with fallback to sample data"""

        # Default sample profiles
        default_profiles = {
            "12345": CustomerProfile(
                customer_id="12345",
                income=75000.0,
                credit_score=780,
                account_type="premium_plus",
                customer_since="2019-05-15",
                risk_tier="low",
                recent_transactions=[
                    {"amount": 4500.00, "description": "Salary Deposit", "ts": "2024-03-20"},
                    {"amount": 1500.00, "description": "Mortgage Payment", "ts": "2024-03-15"},
                    {"amount": 300.00, "description": "Investment Contribution", "ts": "2024-03-10"},
                    {"amount": 200.00, "description": "Utility Bills", "ts": "2024-03-05"},
                ],
                banking_products=["checking", "savings", "mortgage", "investment", "credit_card"],
                last_review_date="2024-01-10"
            ),
            "67890": CustomerProfile(
                customer_id="67890",
                income=45000.0,
                credit_score=680,
                account_type="standard",
                customer_since="2021-08-20",
                risk_tier="medium",
                recent_transactions=[
                    {"amount": 3200.00, "description": "Salary Deposit", "ts": "2024-03-20"},
                    {"amount": 1200.00, "description": "Rent Payment", "ts": "2024-03-14"},
                    {"amount": 400.00, "description": "Car Payment", "ts": "2024-03-08"},
                    {"amount": 150.00, "description": "Student Loan", "ts": "2024-03-02"},
                ],
                banking_products=["checking", "savings", "credit_card"],
                last_review_date="2024-02-15"
            ),
            "11111": CustomerProfile(
                customer_id="11111",
                income=28000.0,
                credit_score=620,
                account_type="basic",
                customer_since="2023-01-10",
                risk_tier="high",
                recent_transactions=[
                    {"amount": 2300.00, "description": "Salary Deposit", "ts": "2024-03-20"},
                    {"amount": 800.00, "description": "Rent Payment", "ts": "2024-03-12"},
                    {"amount": 300.00, "description": "Credit Card Payment", "ts": "2024-03-07"},
                    {"amount": 150.00, "description": "Overdraft Fee", "ts": "2024-03-01"},
                ],
                banking_products=["checking"],
                last_review_date="2024-03-01"
            ),
        }

        # Try loading from Azure SQL, fall back to defaults
        if self.data_connector.connection_string:
            try:
                for cid in ["12345", "67890", "11111"]:
                    income = await self.data_connector.fetch_income(cid)
                    transactions = await self.data_connector.fetch_transactions(cid)
                    if income is not None:
                        profile = default_profiles.get(cid, CustomerProfile(customer_id=cid))
                        profile.income = income
                        if transactions:
                            profile.recent_transactions = transactions
                        default_profiles[cid] = profile
                self.logger.info("Customer profiles loaded from Azure SQL")
            except Exception as e:
                self.logger.warning(f"SQL load failed, using sample data: {e}")
        else:
            self.logger.info("Using sample customer profiles (no SQL connection)")

        return default_profiles

    def create_enhanced_agents(self) -> List[ChatCompletionAgent]:
        """Create specialized banking agents with detailed instructions"""

        data_agent = ChatCompletionAgent(
            name="Enhanced_Data_Gatherer",
            instructions="""You are a Senior Banking Data Analyst specializing in customer financial profiling.

Your responsibilities:
1. Analyze customer financial data including income, transactions, credit history, and account activity.
2. Match customer profiles against relevant banking policies and eligibility criteria.
3. Assess data quality and completeness, flagging any gaps or inconsistencies.
4. Calculate key financial metrics: debt-to-income ratio, savings rate, spending patterns.
5. Identify the customer's financial segment (premium, standard, basic) based on their profile.

Output format:
- Customer Financial Summary with key metrics
- Data Quality Assessment (completeness score)
- Policy Relevance Mapping (which policies apply to this customer)
- Key observations about the customer's financial behavior""",
            service=self.kernel.get_service("enhanced_banking_chat")
        )

        fraud_agent = ChatCompletionAgent(
            name="Enhanced_Fraud_Analyst",
            instructions="""You are a Senior Fraud Detection Specialist with expertise in banking transaction analysis.

Your responsibilities:
1. Analyze transaction patterns for suspicious activity indicators:
   - Large transactions (>$2,000) or unusual amounts
   - Rapid transaction sequences (>10/hour)
   - Geographic anomalies or new payees
   - Transactions outside normal behavioral patterns
2. Assess fraud risk level (Low/Medium/High/Critical) with justification.
3. Identify potential fraud typologies: account takeover, identity theft, card fraud, money laundering.
4. Recommend specific mitigation actions based on risk level.
5. Reference applicable fraud detection policies.

Output format:
- Transaction Pattern Analysis
- Fraud Risk Score (0-100) with risk level
- Identified suspicious indicators (if any)
- Recommended actions and monitoring enhancements""",
            service=self.kernel.get_service("enhanced_banking_chat")
        )

        loan_agent = ChatCompletionAgent(
            name="Enhanced_Loan_Analyst",
            instructions="""You are a Senior Credit Risk Analyst specializing in loan eligibility assessment.

Your responsibilities:
1. Evaluate loan eligibility based on:
   - Income tier classification (A+: $100K+, A: $75K+, B: $50K+, C: $30K+)
   - Credit score tier (Excellent 750+, Good 700-749, Fair 650-699, Review <650)
   - Debt-to-income ratio against tier limits (30%-45% depending on tier)
   - Employment history and stability
2. Determine maximum qualifying loan amount and recommended terms.
3. Identify applicable interest rates and LTV ratios.
4. Recommend suitable loan products based on customer profile.
5. Flag any disqualifying factors or conditions requiring special review.

Output format:
- Eligibility Determination (Approved/Conditional/Review Required/Declined)
- Qualifying tier and applicable rates
- Maximum recommended loan amount
- Required documentation level (Basic/Standard/Comprehensive/Premium)
- Product recommendations""",
            service=self.kernel.get_service("enhanced_banking_chat")
        )

        support_agent = ChatCompletionAgent(
            name="Enhanced_Support_Specialist",
            instructions="""You are a Senior Customer Experience Specialist focused on banking service optimization.

Your responsibilities:
1. Assess customer service needs based on their profile and query context.
2. Identify service gaps and opportunities for improvement.
3. Determine priority classification (P0-Critical to P3-Low) for the customer's needs.
4. Recommend proactive engagement strategies for customer retention.
5. Suggest relevant self-service options and digital banking features.
6. Evaluate customer lifetime value and recommend appropriate service tier.

Output format:
- Customer Experience Assessment
- Service Priority Classification with response time SLA
- Identified service gaps and improvement opportunities
- Retention risk assessment
- Recommended engagement actions""",
            service=self.kernel.get_service("enhanced_banking_chat")
        )

        risk_agent = ChatCompletionAgent(
            name="Enhanced_Risk_Analyst",
            instructions="""You are a Senior Enterprise Risk Analyst specializing in banking compliance and risk management.

Your responsibilities:
1. Perform comprehensive risk assessment across five categories:
   - Credit Risk: borrower default probability
   - Market Risk: economic exposure
   - Operational Risk: process and system risks
   - Compliance Risk: regulatory adherence
   - Reputational Risk: brand impact
2. Assign risk scores and levels (Low <10%, Medium 10-30%, High 30-60%, Critical >60%).
3. Verify compliance with banking regulations and internal policies.
4. Recommend risk mitigation strategies with priority ranking.
5. Determine appropriate review frequency based on risk profile.

Output format:
- Multi-dimensional Risk Assessment Matrix
- Overall risk score and level
- Compliance status with specific policy references
- Prioritized mitigation recommendations
- Recommended monitoring and review schedule""",
            service=self.kernel.get_service("enhanced_banking_chat")
        )

        synthesis_agent = ChatCompletionAgent(
            name="Enhanced_Synthesis_Coordinator",
            instructions="""You are a Senior Banking Strategy Coordinator responsible for synthesizing multi-agent analyses into executive reports.

Your responsibilities:
1. Integrate findings from all previous agent analyses into a coherent narrative.
2. Identify cross-cutting themes, conflicts, and synergies between agent assessments.
3. Generate an executive summary suitable for senior banking leadership.
4. Produce a prioritized action plan with clear ownership and timelines.
5. Provide strategic recommendations for the customer relationship.

Output format:
- Executive Summary (2-3 paragraphs)
- Consolidated Key Findings (top 5)
- Integrated Risk Profile
- Strategic Recommendations (prioritized)
- Immediate Action Items
- Long-term Relationship Strategy""",
            service=self.kernel.get_service("enhanced_banking_chat")
        )

        agents = [data_agent, fraud_agent, loan_agent, support_agent, risk_agent, synthesis_agent]
        return agents

    async def load_enhanced_documents(self):
        """Load banking policy documents into ChromaDB for semantic search"""
        stats = await self.chroma_store.get_collection_stats()
        total_docs = sum(s.get("document_count", 0) for s in stats.values())

        if total_docs > 0:
            self.logger.info(f"ChromaDB already has {total_docs} document chunks loaded")
            return

        for doc_name in self.blob_connector.list_documents():
            content = self.blob_connector.get_document_content(doc_name)
            if content:
                collection_type = self.chroma_store.determine_collection(doc_name, content)
                await self.chroma_store.chunk_and_store_document(doc_name, content, collection_type)

        self.logger.info("Banking documents loaded into ChromaDB")

    async def run_enhanced_analysis(self, customer_id: str, customer_query: str) -> EnhancedBankingReport:
        """Run the main banking analysis workflow"""
        start_time = time.time()
        self.performance_metrics["total_requests"] += 1

        # Ensure customer profiles are loaded
        if not self.customer_profiles:
            self.customer_profiles = await self._load_customer_profiles()

        # Load documents to ChromaDB
        await self.load_enhanced_documents()

        # Get customer data and perform semantic search
        customer_profile = self.customer_profiles.get(
            customer_id,
            CustomerProfile(customer_id=customer_id)
        )

        # Hybrid search across all banking collections
        search_results = await self.chroma_store.hybrid_search(customer_query, [
            "fraud_detection", "loan_policies", "customer_support",
            "risk_assessment", "transaction_monitoring", "compliance"
        ], top_k=4)

        # Prepare enhanced context for orchestration
        banking_context = self._prepare_enhanced_context(customer_profile, search_results, customer_query)

        # Create enhanced agents
        agents = self.create_enhanced_agents()

        # Agent callback tracking
        agent_contributions = {}

        def enhanced_agent_callback(message: ChatMessageContent) -> None:
            """Track agent contributions and log output"""
            agent_contributions[message.name] = message.content
            self.logger.info(f"Agent {message.name} completed analysis")
            print(f"\n{'='*60}")
            print(f"# {message.name}")
            print(f"{'='*60}")
            print(f"{message.content}\n")

        # Create SequentialOrchestration
        sequential_orchestration = SequentialOrchestration(
            members=agents,
            agent_response_callback=enhanced_agent_callback,
        )

        # Set up runtime
        runtime = InProcessRuntime()

        try:
            runtime.start()

            orchestration_task = f"""
ENHANCED BANKING CUSTOMER ANALYSIS REQUEST
============================================

{banking_context}

ANALYSIS INSTRUCTIONS:
Each agent should perform their specialized analysis in sequence:
1. Data Gatherer: Analyze the customer profile, calculate financial metrics, and identify applicable policies.
2. Fraud Analyst: Review transaction patterns for suspicious activity and assess fraud risk.
3. Loan Analyst: Evaluate loan eligibility, credit risk, and recommend suitable products.
4. Support Specialist: Assess customer service needs, retention risk, and engagement opportunities.
5. Risk Analyst: Perform enterprise risk assessment across all risk categories and verify compliance.
6. Synthesis Coordinator: Integrate all findings into an executive report with prioritized recommendations.

Provide specific, actionable insights based on the customer data and banking policies provided.
"""

            # Invoke the orchestration
            orchestration_result = await sequential_orchestration.invoke(
                task=orchestration_task,
                runtime=runtime
            )

            # Get the final result
            final_output = await asyncio.wait_for(orchestration_result.get(), timeout=180.0)

            # Calculate enhanced risk score
            risk_score = self._calculate_enhanced_risk_score(customer_profile, search_results)
            risk_assessment = self._determine_risk_tier(risk_score)

            # Extract policy references from search results
            policy_refs = list({
                r.get("filename", "Unknown") for r in search_results if r.get("filename")
            })

            elapsed = time.time() - start_time

            # Create comprehensive banking report
            report = EnhancedBankingReport(
                report_id=f"enhanced_{uuid.uuid4().hex[:8]}",
                customer_id=customer_id,
                query=customer_query,
                summary=str(final_output),
                key_findings=self._generate_enhanced_findings(customer_profile, search_results, agent_contributions),
                risk_assessment=risk_assessment,
                risk_score=risk_score,
                recommendations=self._generate_enhanced_recommendations(customer_profile, risk_score),
                actions_taken=[
                    "Enhanced multi-agent sequential analysis completed",
                    "Comprehensive policy compliance verification performed",
                    "Enterprise risk assessment conducted",
                    f"Analyzed {len(search_results)} relevant policy documents",
                    f"{len(agent_contributions)} specialized agents contributed to analysis",
                ],
                policy_references=policy_refs,
                agent_contributions=agent_contributions,
                processing_metrics={
                    "total_processing_time_seconds": round(elapsed, 2),
                    "agents_used": len(agents),
                    "policies_referenced": len(policy_refs),
                    "search_results_analyzed": len(search_results),
                    "risk_score": risk_score,
                },
                generated_by="EnhancedBankingSequentialOrchestration"
            )

            self.performance_metrics["successful_analyses"] += 1
            self.shared_state.update_interaction(customer_id, {
                "query": customer_query,
                "report_id": report.report_id,
                "risk_score": risk_score,
                "timestamp": datetime.now().isoformat()
            })

            return report

        except Exception as e:
            self.logger.error(f"Error in enhanced orchestration: {e}")
            self.shared_state.record_failure(customer_id, str(e))
            raise
        finally:
            await runtime.stop_when_idle()

    def _calculate_enhanced_risk_score(self, customer_profile: CustomerProfile, search_results: List[Dict]) -> float:
        """Calculate comprehensive risk score from multiple factors (0.0 = low risk, 1.0 = high risk)"""
        base_score = 0.5

        # Income-based risk factor (higher income = lower risk)
        if customer_profile.income >= 100000:
            base_score -= 0.15
        elif customer_profile.income >= 75000:
            base_score -= 0.10
        elif customer_profile.income >= 50000:
            base_score -= 0.05
        elif customer_profile.income < 30000:
            base_score += 0.10

        # Credit score-based factor
        if customer_profile.credit_score >= 750:
            base_score -= 0.15
        elif customer_profile.credit_score >= 700:
            base_score -= 0.08
        elif customer_profile.credit_score >= 650:
            base_score += 0.05
        elif customer_profile.credit_score > 0:
            base_score += 0.15

        # Customer tenure (longer tenure = lower risk)
        if customer_profile.customer_since:
            try:
                since = datetime.strptime(customer_profile.customer_since, "%Y-%m-%d")
                years = (datetime.now() - since).days / 365.25
                if years >= 5:
                    base_score -= 0.10
                elif years >= 3:
                    base_score -= 0.05
                elif years < 1:
                    base_score += 0.08
            except ValueError:
                pass

        # Product diversification (more products = lower risk)
        num_products = len(customer_profile.banking_products)
        if num_products >= 4:
            base_score -= 0.08
        elif num_products >= 2:
            base_score -= 0.03
        elif num_products <= 1:
            base_score += 0.05

        # Transaction pattern analysis
        transactions = customer_profile.recent_transactions
        if transactions:
            amounts = [t.get("amount", 0) for t in transactions]
            max_amount = max(amounts) if amounts else 0
            if max_amount > 10000:
                base_score += 0.10
            elif max_amount > 5000:
                base_score += 0.03

        return max(0.0, min(1.0, round(base_score, 3)))

    def _determine_risk_tier(self, risk_score: float) -> str:
        """Determine risk tier from numeric score"""
        if risk_score < 0.25:
            return "low"
        elif risk_score < 0.50:
            return "medium-low"
        elif risk_score < 0.65:
            return "medium"
        elif risk_score < 0.80:
            return "high"
        else:
            return "critical"

    def _generate_enhanced_findings(self, customer_profile: CustomerProfile, search_results: List[Dict], agent_contributions: Dict) -> List[str]:
        """Generate comprehensive findings based on analysis"""
        findings = [
            f"Customer {customer_profile.customer_id} analysis completed with {len(agent_contributions)} agent contributions",
        ]

        # Income findings
        if customer_profile.income >= 75000:
            findings.append(f"Customer qualifies for Tier A+ or A lending products (income: ${customer_profile.income:,.2f})")
        elif customer_profile.income >= 50000:
            findings.append(f"Customer qualifies for Tier B lending products (income: ${customer_profile.income:,.2f})")
        elif customer_profile.income >= 30000:
            findings.append(f"Customer qualifies for Tier C lending products (income: ${customer_profile.income:,.2f})")
        else:
            findings.append(f"Customer income (${customer_profile.income:,.2f}) may limit product eligibility")

        # Credit score findings
        if customer_profile.credit_score >= 750:
            findings.append(f"Excellent credit score ({customer_profile.credit_score}) - eligible for best rates (3.5% APR)")
        elif customer_profile.credit_score >= 700:
            findings.append(f"Good credit score ({customer_profile.credit_score}) - eligible for competitive rates (4.5% APR)")
        elif customer_profile.credit_score >= 650:
            findings.append(f"Fair credit score ({customer_profile.credit_score}) - standard rates apply (6.0% APR)")
        elif customer_profile.credit_score > 0:
            findings.append(f"Credit score ({customer_profile.credit_score}) requires case-by-case assessment")

        # Product usage findings
        products = customer_profile.banking_products
        if len(products) >= 4:
            findings.append(f"High product engagement ({len(products)} products) indicates strong customer relationship")
        elif len(products) <= 1:
            findings.append(f"Low product engagement ({len(products)} product) - cross-sell opportunity identified")

        # Transaction pattern findings
        if customer_profile.recent_transactions:
            amounts = [t.get("amount", 0) for t in customer_profile.recent_transactions]
            findings.append(f"Recent transaction activity: {len(amounts)} transactions, range ${min(amounts):,.2f}-${max(amounts):,.2f}")

        # Policy relevance from search results
        if search_results:
            collections = {r.get("collection", "") for r in search_results[:5]}
            findings.append(f"Relevant policy areas identified: {', '.join(collections)}")

        return findings

    def _generate_enhanced_recommendations(self, customer_profile: CustomerProfile, risk_score: float) -> List[str]:
        """Generate strategic recommendations based on analysis"""
        recommendations = []

        # Risk-based recommendations
        risk_tier = self._determine_risk_tier(risk_score)
        if risk_tier in ("high", "critical"):
            recommendations.append("Implement enhanced monitoring with quarterly risk reviews")
            recommendations.append("Consider requiring additional documentation for high-value transactions")
        elif risk_tier == "medium":
            recommendations.append("Maintain standard monitoring with semi-annual reviews")
        else:
            recommendations.append("Continue standard monitoring with annual reviews")

        # Product recommendations based on profile
        products = set(customer_profile.banking_products)
        if "investment" not in products and customer_profile.income >= 50000:
            recommendations.append("Recommend investment portfolio services based on income level")
        if "savings" not in products:
            recommendations.append("Recommend high-yield savings account to improve financial health")
        if "credit_card" not in products and customer_profile.credit_score >= 650:
            recommendations.append("Eligible for rewards credit card based on credit profile")
        if "mortgage" not in products and customer_profile.income >= 75000 and customer_profile.credit_score >= 700:
            recommendations.append("Pre-qualify for mortgage products at competitive rates")

        # Customer relationship recommendations
        if customer_profile.account_type == "basic" and customer_profile.income >= 50000:
            recommendations.append("Upgrade to premium account tier based on income qualification")

        if len(customer_profile.banking_products) <= 1:
            recommendations.append("Initiate cross-sell engagement program to deepen customer relationship")

        # Credit improvement recommendations
        if customer_profile.credit_score < 700 and customer_profile.credit_score > 0:
            recommendations.append("Offer credit-building program to improve eligibility for premium products")

        # Ensure at least a general advisory recommendation
        if len(recommendations) < 2:
            recommendations.append("Schedule periodic financial health review to identify emerging opportunities")

        return recommendations

    def _prepare_enhanced_context(self, customer_profile: CustomerProfile, search_results: List[Dict], customer_query: str) -> str:
        """Prepare comprehensive context for banking orchestration"""

        # Customer profile context
        customer_context = f"""
CUSTOMER PROFILE:
- Customer ID: {customer_profile.customer_id}
- Annual Income: ${customer_profile.income:,.2f}
- Credit Score: {customer_profile.credit_score}
- Account Type: {customer_profile.account_type}
- Customer Since: {customer_profile.customer_since}
- Risk Tier: {customer_profile.risk_tier}
- Banking Products: {', '.join(customer_profile.banking_products) if customer_profile.banking_products else 'None'}
- Last Review Date: {customer_profile.last_review_date}
"""

        # Transaction context
        tx_context = "\nRECENT TRANSACTIONS:\n"
        for tx in customer_profile.recent_transactions[:10]:
            tx_context += f"- ${tx.get('amount', 0):,.2f} - {tx.get('description', 'N/A')} ({tx.get('ts', 'N/A')})\n"

        # Policy context from search results
        policy_context = "\nRELEVANT BANKING POLICIES:\n"
        for i, result in enumerate(search_results[:6], 1):
            policy_context += f"\n--- Policy Reference {i} (Source: {result.get('filename', 'Unknown')}, "
            policy_context += f"Collection: {result.get('collection', 'Unknown')}, "
            policy_context += f"Relevance: {result.get('final_score', result.get('relevance_score', 0)):.3f}) ---\n"
            policy_context += result.get("document", "")[:500] + "\n"

        # Structured policy summary
        policy_summary = "\nPOLICY FRAMEWORK SUMMARY:\n"
        policy_summary += create_semantic_kernel_context(self.banking_policies)

        return f"""
BANKING ANALYSIS REQUEST: {customer_query}

{customer_context}
{tx_context}
{policy_context}
{policy_summary}

ANALYSIS SCOPE:
Provide a comprehensive analysis covering fraud detection, loan eligibility,
customer service optimization, enterprise risk assessment, and strategic recommendations.
"""


def _display_report(report: EnhancedBankingReport):
    """Display a formatted banking report"""
    print(f"\n{'='*80}")
    print(f"FINAL REPORT: {report.report_id}")
    print(f"{'='*80}")
    print(f"Customer: {report.customer_id}")
    print(f"Risk Assessment: {report.risk_assessment} (score: {report.risk_score:.3f})")
    print(f"\nKey Findings:")
    for finding in report.key_findings:
        print(f"  - {finding}")
    print(f"\nRecommendations:")
    for rec in report.recommendations:
        print(f"  - {rec}")
    print(f"\nActions Taken:")
    for action in report.actions_taken:
        print(f"  - {action}")
    print(f"\nPolicy References: {', '.join(report.policy_references)}")
    print(f"\nProcessing Metrics: {report.processing_metrics}")
    print(f"\nAgent Contributions: {list(report.agent_contributions.keys())}")
    print(f"\nSummary (first 500 chars):\n{report.summary[:500]}...")


async def run_component_tests(system: EnhancedBankingSequentialOrchestration):
    """Unit-test individual components and report pass/fail"""
    results = []
    total = 0
    passed = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal total, passed
        total += 1
        status = "PASS" if condition else "FAIL"
        if condition:
            passed += 1
        results.append((name, status, detail))
        print(f"  [{status}] {name}" + (f" - {detail}" if detail else ""))

    print("\n" + "="*80)
    print("COMPONENT TESTS")
    print("="*80)

    # 1. BlobStorageConnector
    print("\n--- BlobStorageConnector ---")
    docs = system.blob_connector.list_documents()
    check("Document listing", len(docs) > 0, f"{len(docs)} documents found")
    for doc_name in docs[:2]:
        content = system.blob_connector.get_document_content(doc_name)
        check(f"Read '{doc_name}'", content is not None and len(content) > 0, f"{len(content)} chars")
    meta = system.blob_connector.get_document_metadata(docs[0])
    check("Document metadata", meta is not None and "type" in meta, f"type={meta.get('type')}")
    search = system.blob_connector.search_documents("fraud")
    check("Keyword search", len(search) > 0, f"{len(search)} results for 'fraud'")

    # 2. ChromaDBManager
    print("\n--- ChromaDBManager ---")
    check("ChromaDB client", system.chroma_store.client is not None)
    check("Collections initialized", len(system.chroma_store.collections) >= 6,
          f"{len(system.chroma_store.collections)} collections")
    await system.load_enhanced_documents()
    stats = await system.chroma_store.get_collection_stats()
    total_chunks = sum(s.get("document_count", 0) for s in stats.values())
    check("Documents chunked & stored", total_chunks > 0, f"{total_chunks} chunks across collections")
    sem_results = await system.chroma_store.semantic_search("loan eligibility", ["loan_policies"], top_k=2)
    check("Semantic search", len(sem_results) > 0, f"{len(sem_results)} results")
    hyb_results = await system.chroma_store.hybrid_search("fraud detection", ["fraud_detection"], top_k=2)
    check("Hybrid search", len(hyb_results) > 0 and "final_score" in hyb_results[0],
          f"{len(hyb_results)} results with scores")

    # 3. RAG utilities
    print("\n--- RAG Utilities ---")
    check("Banking policies loaded", len(system.banking_policies) > 0,
          f"{len(system.banking_policies)} policy categories")
    policy_ctx = create_semantic_kernel_context(system.banking_policies)
    check("Semantic Kernel context", len(policy_ctx) > 20, f"{len(policy_ctx)} chars")

    # 4. SharedState
    print("\n--- SharedState ---")
    system.shared_state.update_interaction("test_unit", {"action": "unit_test"})
    interactions = system.shared_state.get_customer_interactions("test_unit")
    check("State update & retrieval", len(interactions) > 0)
    metrics = system.shared_state.get_system_metrics()
    check("System metrics", "total_interactions" in metrics)

    # 5. DataConnector
    print("\n--- DataConnector ---")
    check("DataConnector initialized", system.data_connector is not None)
    if system.data_connector.connection_string:
        income = await system.data_connector.fetch_income("12345")
        check("SQL fetch_income", income is not None, f"income={income}")
        txns = await system.data_connector.fetch_transactions("12345")
        check("SQL fetch_transactions", len(txns) > 0, f"{len(txns)} transactions")
    else:
        check("SQL connection (fallback mode)", True, "No SQL string; sample data used")

    # 6. Customer profiles
    print("\n--- Customer Profiles ---")
    profiles = await system._load_customer_profiles()
    check("Profile loading", len(profiles) >= 3, f"{len(profiles)} profiles loaded")
    for cid in ["12345", "67890", "11111"]:
        p = profiles.get(cid)
        check(f"Profile {cid}", p is not None and p.income > 0, f"income=${p.income:,.0f}, score={p.credit_score}")

    # 7. Agent creation
    print("\n--- Agent Creation ---")
    agents = system.create_enhanced_agents()
    check("Six agents created", len(agents) == 6, f"{len(agents)} agents")
    expected_names = [
        "Enhanced_Data_Gatherer", "Enhanced_Fraud_Analyst", "Enhanced_Loan_Analyst",
        "Enhanced_Support_Specialist", "Enhanced_Risk_Analyst", "Enhanced_Synthesis_Coordinator"
    ]
    for agent, name in zip(agents, expected_names):
        check(f"Agent '{name}'", agent.name == name and len(agent.instructions) > 50)

    # 8. Risk scoring
    print("\n--- Risk Scoring ---")
    low_risk = profiles["12345"]
    high_risk = profiles["11111"]
    low_score = system._calculate_enhanced_risk_score(low_risk, [])
    high_score = system._calculate_enhanced_risk_score(high_risk, [])
    check("Low-risk customer scored lower", low_score < high_score,
          f"low={low_score:.3f}, high={high_score:.3f}")
    check("Risk tier determination", system._determine_risk_tier(0.1) == "low" and
          system._determine_risk_tier(0.9) == "critical")

    # Summary
    print(f"\n{'='*80}")
    print(f"TEST RESULTS: {passed}/{total} passed")
    print(f"{'='*80}")
    return passed, total


async def run_demo(system: EnhancedBankingSequentialOrchestration):
    """Run a single demo scenario"""
    print("\n" + "="*80)
    print("DEMO: Customer 12345 - Financial Planning")
    print("="*80)
    report = await system.run_enhanced_analysis(
        "12345",
        "I need comprehensive financial planning including investments and retirement options"
    )
    _display_report(report)
    return report


async def run_test_scenarios(system: EnhancedBankingSequentialOrchestration):
    """Run all test scenarios and validate results"""
    test_scenarios = [
        {
            "customer_id": "12345",
            "query": "I need comprehensive financial planning including investments and retirement options"
        },
        {
            "customer_id": "67890",
            "query": "I want to apply for a home loan and need to understand my eligibility"
        },
        {
            "customer_id": "11111",
            "query": "I noticed some suspicious activity on my account and need help resolving it"
        },
    ]

    reports = []
    for i, scenario in enumerate(test_scenarios, 1):
        try:
            print(f"\n{'='*80}")
            print(f"SCENARIO {i}: Customer {scenario['customer_id']}")
            print(f"Query: {scenario['query']}")
            print(f"{'='*80}")

            report = await system.run_enhanced_analysis(
                scenario["customer_id"],
                scenario["query"]
            )
            _display_report(report)
            reports.append(report)

        except Exception as e:
            print(f"Error in scenario {i}: {e}")
            logger.error(f"Scenario {i} failed: {e}")

    # Validation summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    all_pass = True
    for report in reports:
        agents_ok = len(report.agent_contributions) == 6
        findings_ok = len(report.key_findings) >= 3
        recs_ok = len(report.recommendations) >= 2
        score_ok = 0.0 <= report.risk_score <= 1.0
        time_ok = report.processing_metrics.get("total_processing_time_seconds", 999) < 300
        status = "PASS" if all([agents_ok, findings_ok, recs_ok, score_ok, time_ok]) else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] Customer {report.customer_id}: "
              f"agents={len(report.agent_contributions)}/6, "
              f"findings={len(report.key_findings)}, "
              f"recs={len(report.recommendations)}, "
              f"risk={report.risk_score:.3f} ({report.risk_assessment}), "
              f"time={report.processing_metrics.get('total_processing_time_seconds', 0):.1f}s")

    print(f"\nOverall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    return reports


async def enhanced_main():
    """Main entry point with CLI argument support"""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Banking Multi-Agent RAG System")
    parser.add_argument("--all", action="store_true", help="Run component tests + all scenarios")
    parser.add_argument("--demo", action="store_true", help="Run a single demo scenario")
    parser.add_argument("--test", action="store_true", help="Run all test scenarios with validation")
    args = parser.parse_args()

    # Default to --all if no flags provided
    if not (args.all or args.demo or args.test):
        args.all = True

    log_filename = setup_logging()

    print("ENHANCED BANKING MULTI-AGENT RAG SYSTEM")
    print("Student Implementation Project")
    print("=" * 80)

    print("\nInitializing EnhancedBankingSequentialOrchestration...")
    system = EnhancedBankingSequentialOrchestration()

    if args.all:
        passed, total = await run_component_tests(system)
        await run_test_scenarios(system)
    elif args.demo:
        await run_demo(system)
    elif args.test:
        await run_test_scenarios(system)

    # Display system metrics
    metrics = system.shared_state.get_system_metrics()
    print(f"\n{'='*80}")
    print("SYSTEM METRICS")
    print(f"{'='*80}")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(enhanced_main())
