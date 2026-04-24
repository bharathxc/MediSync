"""
MediSync AI — Agentic RAG Tools
RAG retriever and CRM lookup tools for the LangGraph agent.
"""
import json
import chromadb
from langchain_core.tools import tool
from config import CHROMA_DB_DIR, CRM_DATA_PATH, TOP_K_RESULTS


def _get_collection():
    """Get the ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return client.get_collection("medisync_plans")


def _load_crm_data() -> list[dict]:
    """Load CRM data from JSON file."""
    with open(CRM_DATA_PATH, 'r') as f:
        data = json.load(f)
    return data["members"]


@tool
def get_policy_info(query: str) -> str:
    """Search the Optum health plan knowledge base for policy information,
    drug coverage, copays, deductibles, and benefit details.
    Use this tool when a user asks about plan benefits, coverage, costs, or procedures.

    Args:
        query: The search query about health plan benefits or coverage.
    """
    try:
        collection = _get_collection()

        results = collection.query(
            query_texts=[query],
            n_results=TOP_K_RESULTS,
        )

        if not results['documents'][0]:
            return "No relevant policy information found for this query."

        formatted = []
        for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            source = meta.get('source', 'Unknown')
            page = meta.get('page', '?')
            plan_type = meta.get('plan_type', 'Unknown')
            formatted.append(
                f"[Source: {source} | Page {page} | Plan: {plan_type}]\n{doc}"
            )

        return "\n\n---\n\n".join(formatted)

    except Exception as e:
        return f"Error searching policy database: {str(e)}"


@tool
def get_member_details(identifier: str) -> str:
    """Look up a member's profile in the CRM system by name or member ID.
    Use this when a user asks about 'my' plan, 'my' coverage, or needs personalized information.

    Args:
        identifier: The member's name (partial or full) or member ID to search for.
    """
    try:
        members = _load_crm_data()

        for member in members:
            if member['member_id'].lower() == identifier.lower():
                return json.dumps(member, indent=2)

        identifier_lower = identifier.lower()
        matches = []
        for member in members:
            if identifier_lower in member['name'].lower():
                matches.append(member)

        if len(matches) == 1:
            return json.dumps(matches[0], indent=2)
        elif len(matches) > 1:
            names = [f"- {m['name']} (ID: {m['member_id']}, Plan: {m['plan_type']})"
                     for m in matches]
            return "Multiple members found. Please specify:\n" + "\n".join(names)
        else:
            return (f"No member found matching '{identifier}'. "
                    "Please verify the name or member ID and try again.")

    except Exception as e:
        return f"Error accessing CRM system: {str(e)}"


TOOLS = [get_policy_info, get_member_details]