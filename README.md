# 🛠️ Autonomous Field Service AI Coordinator Agent

Production-ready field service automation system built with **LangGraph**, **LangChain**, **Google Gemini**, **Hugging Face**, and **Streamlit**.

## 📌 Architecture Highlights
- **State Machine Orchestration:** Uses `LangGraph` for multi-step routing, automated checks, and Human-in-the-Loop approval handling.
- **Structured LLM Extraction:** Uses Gemini API with Pydantic schemas to output valid JSON entities from noisy email text.
- **Intent & Skill Pipeline:** Integrated with Hugging Face `transformers` and `datasets` for modular intent classification.
- **Admin Review UI:** Streamlit dashboard for dispatcher review, manual overrides, and instant log inspectability.

---

## 🚀 Local Setup & Running Instructions

1. **Clone Repository & Navigate:**
   ```bash
   git clone [https://github.com/your-org/field-service-ai-agent.git](https://github.com/your-org/field-service-ai-agent.git)
   cd field-service-ai-agent