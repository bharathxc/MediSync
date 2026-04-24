"""
MediSync AI — Unified Configuration
Combines RAG Agent and Clinical NLP Safeguard settings.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── Base Paths ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "knowledge_base")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
CRM_DATA_PATH = os.path.join(BASE_DIR, "crm_data.json")
DATA_DIR = os.path.join(BASE_DIR, "data")

# ─── LLM Backend Selection ────────────────────────────────────────────────
# Options: "ollama" or "google"
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")

# Ollama Configuration (local — FREE)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Google Gemini Configuration (cloud — requires API key)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")

# Shared Settings
LLM_TEMPERATURE = 0.3
LLM_TEMPERATURE_DETERMINISTIC = 0.0  # For de-identification (want exact output)

# ─── RAG Configuration ────────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 5

# ─── NLP Configuration (Presidio/spaCy) ─────────────────────────────────
NLP_ENGINE = "spacy"
SPACY_MODEL = "en_core_web_lg"

# ─── LLM Initialization Helpers ──────────────────────────────────────────────
def get_llm():
    """Initialize the LLM based on configured backend."""
    if LLM_BACKEND.lower() == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=GOOGLE_MODEL,
            temperature=LLM_TEMPERATURE,
            google_api_key=GOOGLE_API_KEY,
            convert_system_message_to_human=True,
        )
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=OLLAMA_MODEL,
            temperature=LLM_TEMPERATURE,
            base_url=OLLAMA_BASE_URL,
        )


def get_llm_deterministic():
    """Initialize LLM with deterministic settings for de-identification."""
    if LLM_BACKEND.lower() == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=GOOGLE_MODEL,
            temperature=LLM_TEMPERATURE_DETERMINISTIC,
            google_api_key=GOOGLE_API_KEY,
        )
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=OLLAMA_MODEL,
            temperature=LLM_TEMPERATURE_DETERMINISTIC,
            base_url=OLLAMA_BASE_URL,
        )


# ─── System Prompts ────────────────────────────────────────────────────
AGENT_SYSTEM_PROMPT = """You are MediSync AI, an intelligent healthcare benefits navigator developed for Optum.
Your role is to help members understand their health insurance coverage, prescription drug benefits,
copays, deductibles, and plan-specific details.

CRITICAL RULES:
1. ONLY answer using the provided context documents. Do NOT make up information.
2. If the answer is not in the provided documents, respond with:
   "I do not have enough information to confirm your coverage details. Please contact Optum support
   at 1-800-555-0199 (Basic Plan) or 1-800-555-0200 (Premium Plan) for assistance."
3. Always specify which plan (Basic or Premium) the information applies to.
4. When discussing costs, always include the exact dollar amounts from the documents.
5. If a member has been identified, personalize your response using their plan type.
6. Never disclose sensitive member information (SSN, full DOB, etc.) in responses.
7. Be empathetic and professional — members may be dealing with health concerns.
8. If asked about services that require prior authorization, clearly state that requirement.

RESPONSE FORMAT:
- Use clear, organized formatting with bullet points for multiple items.
- Include relevant plan details (Plan ID, copay amounts, coverage percentages).
- End with a helpful follow-up suggestion when appropriate.
"""

AGENT_SUPERVISOR_PROMPT = """You are the routing supervisor for MediSync AI. Analyze the user's message and determine
which tools are needed to answer their question.

ROUTING LOGIC:
- If the user asks about "my" plan, "my" coverage, or refers to themselves as a member → the member
  must be identified first. Use get_member_details if not already identified.
- If the user asks about plan benefits, drug coverage, copays, or deductibles → use get_policy_info
  to search the knowledge base.
- If the user asks a general greeting or non-health question → respond directly without tools.

Always think step by step about what information is needed before responding.
"""

NLP_DEID_SYSTEM_PROMPT = """You are a specialized Clinical Data De-identification AI.
You receive clinical notes where a traditional NLP engine (like Presidio/spaCy) has flagged specific entities as PII/PHI. Traditional engines often suffer from false positives and may redact medical terminology like "Down's syndrome" or hospital names relevant to the condition.
Your task is to refine the redaction.

CRITICAL INSTRUCTIONS:
1. Review the provided text and the flagged entities.
2. Remove any real Personally Identifiable Information (Names, Dates of Birth, Phone Numbers, Addresses, SSNs).
3. Do NOT redact medical conditions (e.g., "Parkinson's disease", "Tommy John surgery").
4. Output the refined text keeping the medical context intact but the PII redacted using placeholders like [REDACTED_NAME], [REDACTED_DATE].
5. Provide a brief 1-sentence reasoning at the end prefixed with 'REASONING:'.
"""