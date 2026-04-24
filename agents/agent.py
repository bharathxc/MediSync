"""
MediSync AI — LangGraph Agent Orchestration
Defines the agentic workflow: Supervisor -> Member Lookup -> RAG -> Response.
Supports both Ollama (local) and Google Gemini (cloud) LLM backends.
"""
import operator
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from config import (
    LLM_BACKEND,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    GOOGLE_API_KEY,
    GOOGLE_MODEL,
    AGENT_SYSTEM_PROMPT,
)
from agents.guardrails import mask_pii
from agents.tools import TOOLS


class AgentState(TypedDict):
    """Tracks conversation state through the agent graph."""
    messages: Annotated[list[BaseMessage], operator.add]
    member_info: Optional[dict]
    plan_type: Optional[str]
    pii_detected: Optional[list[dict]]


def get_llm():
    """Initialize the LLM based on configured backend."""
    if LLM_BACKEND == "google":
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


def supervisor_node(state: AgentState) -> dict:
    """The Supervisor analyzes the user's message and decides whether to call tools."""
    messages = state["messages"]
    member_info = state.get("member_info")
    plan_type = state.get("plan_type")

    pii_detections = []
    last_msg = messages[-1]
    if isinstance(last_msg, HumanMessage):
        masked_content, pii_detections = mask_pii(last_msg.content)
        if pii_detections:
            messages = messages[:-1] + [HumanMessage(content=masked_content)]

    system_content = AGENT_SYSTEM_PROMPT
    if member_info:
        system_content += f"\n\nCurrent Member Context:\n"
        system_content += f"- Name: {member_info.get('name', 'Unknown')}\n"
        system_content += f"- Plan: {plan_type or member_info.get('plan_type', 'Unknown')}\n"
        system_content += f"- Member ID: {member_info.get('member_id', 'Unknown')}\n"
        system_content += f"- Status: {member_info.get('status', 'Unknown')}\n"

    full_messages = [SystemMessage(content=system_content)] + messages

    llm = get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)
    response = llm_with_tools.invoke(full_messages)

    return {
        "messages": [response],
        "pii_detected": pii_detections if pii_detections else None,
    }


def tool_executor_node(state: AgentState) -> dict:
    """Execute tools called by the supervisor."""
    tool_node = ToolNode(TOOLS)
    result = tool_node.invoke(state)
    return result


def response_node(state: AgentState) -> dict:
    """Generate the final grounded response after tool results are available."""
    import json

    messages = state["messages"]
    member_info = state.get("member_info")
    plan_type = state.get("plan_type")

    new_member_info = member_info
    new_plan_type = plan_type

    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            try:
                parsed = json.loads(msg.content)
                if isinstance(parsed, dict) and "member_id" in parsed:
                    new_member_info = parsed
                    new_plan_type = parsed.get("plan_type")
                    break
            except (json.JSONDecodeError, TypeError):
                pass

    system_content = AGENT_SYSTEM_PROMPT
    if new_member_info:
        system_content += f"\n\nIdentified Member:\n"
        system_content += f"- Name: {new_member_info.get('name')}\n"
        system_content += f"- Plan: {new_plan_type}\n"
        system_content += f"- Status: {new_member_info.get('status')}\n"

    full_messages = [SystemMessage(content=system_content)] + messages

    llm = get_llm()
    response = llm.invoke(full_messages)

    return {
        "messages": [response],
        "member_info": new_member_info,
        "plan_type": new_plan_type,
    }


def should_use_tools(state: AgentState) -> str:
    """Determine if the supervisor wants to use tools or respond directly."""
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "respond"


def after_tools(state: AgentState) -> str:
    """After tool execution, always go to the response node for synthesis."""
    return "respond"


def build_agent_graph():
    """Construct the LangGraph agent workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("tools", tool_executor_node)
    workflow.add_node("respond", response_node)

    workflow.set_entry_point("supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        should_use_tools,
        {
            "tools": "tools",
            "respond": END,
        }
    )

    workflow.add_edge("tools", "respond")
    workflow.add_edge("respond", END)

    return workflow.compile()


class MediSyncAgent:
    """High-level interface for interacting with the MediSync AI agent."""

    def __init__(self):
        self.graph = build_agent_graph()
        self.state = {
            "messages": [],
            "member_info": None,
            "plan_type": None,
            "pii_detected": None,
        }

    def set_member(self, member_info: dict):
        """Pre-set the active member (from sidebar selection)."""
        self.state["member_info"] = member_info
        self.state["plan_type"] = member_info.get("plan_type")

    def clear_member(self):
        """Clear the active member."""
        self.state["member_info"] = None
        self.state["plan_type"] = None

    def chat(self, user_message: str) -> str:
        """Process a user message through the agent graph. Returns the AI's response text."""
        input_state = {
            "messages": self.state["messages"] + [HumanMessage(content=user_message)],
            "member_info": self.state.get("member_info"),
            "plan_type": self.state.get("plan_type"),
            "pii_detected": None,
        }

        result = self.graph.invoke(input_state)

        self.state["messages"] = result["messages"]
        if result.get("member_info"):
            self.state["member_info"] = result["member_info"]
        if result.get("plan_type"):
            self.state["plan_type"] = result["plan_type"]

        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content

        return "I apologize, but I was unable to generate a response. Please try again."

    def reset(self):
        """Reset conversation history while keeping member context."""
        member = self.state.get("member_info")
        plan = self.state.get("plan_type")
        self.state = {
            "messages": [],
            "member_info": member,
            "plan_type": plan,
            "pii_detected": None,
        }