"""
streamlit_app.py
Streamlit Web Application & Admin Control Panel.
Serves as the UI for Streamlit Cloud deployment.
"""

import streamlit as st
import uuid
import json
from data_loader import load_and_prepare_hf_dataset
from agent import build_field_service_agent

st.set_page_config(
    page_title="Field Service AI Coordinator",
    page_icon="🛠️",
    layout="wide"
)

st.title("🛠️ Autonomous Field Service AI Agent")
st.caption("Production-Grade Multi-Step Orchestrator with LangGraph, Gemini, and Human-in-the-Loop Safeguards")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("🔑 Credentials & Settings")
    gemini_api_key = st.text_input("Google Gemini API Key", type="password")
    st.markdown("---")
    st.subheader("📊 Hugging Face Dataset Tool")
    if st.button("Download & Load HF Dataset"):
        with st.spinner("Fetching field service ticket data..."):
            ds, labels = load_and_prepare_hf_dataset()
            st.success(f"Dataset Loaded! {len(ds['train'])} train examples.")
            st.json(labels)

# Initialize Session State
if "tickets" not in st.session_state:
    st.session_state.tickets = []

if not gemini_api_key:
    st.warning("⚠️ Please enter your Google Gemini API Key in the sidebar to run the agent.")
    st.stop()

# Build Graph Agent
agent = build_field_service_agent()

# --- Section 1: Inbound Ticket Simulator ---
st.subheader("📥 1. Inbound Request Processing")

sample_emails = [
    "Our commercial freezer unit stopped cooling at the Downtown branch. Stock is at risk. Send someone ASAP!",
    "Hi team, can we schedule a routine check for our office HVAC unit next week Thursday?",
    "Main circuit breaker tripped in the industrial bay. Sparks were visible. Urgent assistance required."
]

selected_sample = st.selectbox("Or select a predefined sample email:", ["-- Custom --"] + sample_emails)

if selected_sample != "-- Custom --":
    email_input = st.text_area("Customer Email Body:", value=selected_sample, height=100)
else:
    email_input = st.text_area("Customer Email Body:", placeholder="Type incoming customer request here...", height=100)

if st.button("🚀 Submit to AI Agent Pipeline", type="primary"):
    if not email_input.strip():
        st.error("Please enter email content.")
    else:
        with st.spinner("Processing workflow through LangGraph state machine..."):
            initial_state = {
                "ticket_id": str(uuid.uuid4())[:8],
                "email_text": email_input,
                "extracted_data": None,
                "matched_tech": None,
                "drafted_response": None,
                "status": "INIT",
                "escalation_reason": None,
                "api_key": gemini_api_key
            }
            
            # Execute Agent Graph
            final_state = agent.invoke(initial_state)
            st.session_state.tickets.append(final_state)
            st.success("Workflow Step Complete!")

# --- Section 2: Admin Dashboard & Approval Queue ---
st.markdown("---")
st.subheader("🖥️ 2. Dispatcher Control Panel & Approval Queue")

if not st.session_state.tickets:
    st.info("No active tickets processed yet. Submit an email above.")
else:
    for idx, ticket in enumerate(reversed(st.session_state.tickets)):
        with st.expander(f"Ticket #{ticket['ticket_id']} — Status: {ticket['status']}", expanded=True):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**Original Email:**")
                st.info(ticket["email_text"])
                
                if ticket.get("extracted_data"):
                    st.markdown("**Extracted Entities (Gemini Structured Output):**")
                    st.json(ticket["extracted_data"])
            
            with col2:
                if ticket.get("matched_tech"):
                    st.markdown("**Matched Technician:**")
                    st.success(f"👤 {ticket['matched_tech']['name']} | 📅 {ticket['matched_tech']['available_slot']}")
                
                # Human-in-the-Loop Gate Logic
                if ticket["status"] == "NEEDS_HUMAN_APPROVAL":
                    st.warning(f"🚨 **Escalation Reason:** {ticket.get('escalation_reason')}")
                    
                    if st.button(f"✅ Approve & Dispatch Ticket #{ticket['ticket_id']}", key=f"app_{idx}"):
                        ticket["status"] = "APPROVED"
                        # Trigger drafting node manually after approval
                        from model import draft_customer_response
                        ticket["drafted_response"] = draft_customer_response(
                            api_key=gemini_api_key,
                            extracted_info=ticket["extracted_data"],
                            tech_name=ticket["matched_tech"]["name"],
                            slot=ticket["matched_tech"]["available_slot"]
                        )
                        st.rerun()
                        
                elif ticket["status"] in ["APPROVED", "AUTO_APPROVED"]:
                    st.markdown("**Drafted Response Email:**")
                    st.text_area("Email Content", value=ticket.get("drafted_response", ""), height=120, key=f"txt_{idx}")
                    st.success("✅ Ready for dispatch / Sent to CRM.")