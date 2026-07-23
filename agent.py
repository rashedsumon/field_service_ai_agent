"""
agent.py
LangGraph Orchestration Engine: Manages the multi-step state machine,
technician routing, auto-escalation triggers, and human approval steps.
"""

import uuid
from typing import TypedDict, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from model import get_gemini_extractor, draft_customer_response

# Define Agent State
class ServiceAgentState(TypedDict):
    ticket_id: str
    email_text: str
    extracted_data: Optional[Dict[str, Any]]
    matched_tech: Optional[Dict[str, Any]]
    drafted_response: Optional[str]
    status: str  # "EXTRACTED", "NEEDS_HUMAN_APPROVAL", "APPROVED", "DISPATCHED", "REJECTED"
    escalation_reason: Optional[str]
    api_key: str

# Sample Available Technicians Database
TECHNICIANS = [
    {"name": "Sarah Jenkins", "skills": ["Refrigeration", "Chillers", "HVAC"], "location": "Downtown", "available_slot": "Today at 10:30 AM"},
    {"name": "Marcus Vance", "skills": ["High Voltage", "Electrical", "Commercial Electrical"], "location": "Industrial Park", "available_slot": "Today at 1:00 PM"},
    {"name": "Alex Rivera", "skills": ["Fire Safety", "Inspection", "Plumbing"], "location": "Suburbs", "available_slot": "Tomorrow at 9:00 AM"}
]

# --- Node Functions ---

def extract_entities_node(state: ServiceAgentState) -> Dict[str, Any]:
    """Node: Uses Gemini to extract structured parameters from raw text."""
    extractor = get_gemini_extractor(state["api_key"])
    res = extractor.invoke({"email_text": state["email_text"]})
    extracted_dict = res.model_dump()
    
    return {
        "extracted_data": extracted_dict,
        "status": "EXTRACTED"
    }

def match_technician_node(state: ServiceAgentState) -> Dict[str, Any]:
    """Node: Matches required ticket skills with available technician roster."""
    data = state["extracted_data"] or {}
    req_skills = [s.lower() for s in data.get("required_skills", [])]
    
    best_match = TECHNICIANS[0]  # Fallback default
    for tech in TECHNICIANS:
        tech_skills = [s.lower() for s in tech["skills"]]
        if any(skill in tech_skills for skill in req_skills):
            best_match = tech
            break
            
    return {"matched_tech": best_match}

def evaluate_escalation_node(state: ServiceAgentState) -> Dict[str, Any]:
    """Node: Applies business rules to check if human approval is required."""
    urgency = state["extracted_data"].get("urgency", "LOW").upper()
    
    if urgency in ["HIGH", "CRITICAL"]:
        return {
            "status": "NEEDS_HUMAN_APPROVAL",
            "escalation_reason": f"High urgency ticket flagged ({urgency} priority)."
        }
    
    return {"status": "AUTO_APPROVED"}

def draft_response_node(state: ServiceAgentState) -> Dict[str, Any]:
    """Node: Generates the email response draft."""
    draft = draft_customer_response(
        api_key=state["api_key"],
        extracted_info=state["extracted_data"],
        tech_name=state["matched_tech"]["name"],
        slot=state["matched_tech"]["available_slot"]
    )
    return {"drafted_response": draft}

# --- Conditional Edge Router ---

def route_approval(state: ServiceAgentState) -> str:
    """Routes state based on whether human intervention is needed."""
    if state["status"] == "NEEDS_HUMAN_APPROVAL":
        return "human_approval_gate"
    return "draft_response"

def human_approval_gate(state: ServiceAgentState) -> Dict[str, Any]:
    """Pass-through gate waiting for Streamlit Admin intervention."""
    return {}

# --- Build LangGraph State Machine ---

def build_field_service_agent():
    builder = StateGraph(ServiceAgentState)
    
    # Add Nodes
    builder.add_node("extract_entities", extract_entities_node)
    builder.add_node("match_technician", match_technician_node)
    builder.add_node("evaluate_escalation", evaluate_escalation_node)
    builder.add_node("human_approval_gate", human_approval_gate)
    builder.add_node("draft_response", draft_response_node)
    
    # Add Edges
    builder.add_edge(START, "extract_entities")
    builder.add_edge("extract_entities", "match_technician")
    builder.add_edge("match_technician", "evaluate_escalation")
    
    builder.add_conditional_edges(
        "evaluate_escalation",
        route_approval,
        {
            "human_approval_gate": "human_approval_gate",
            "draft_response": "draft_response"
        }
    )
    
    builder.add_edge("draft_response", END)
    builder.add_edge("human_approval_gate", END)
    
    return builder.compile()