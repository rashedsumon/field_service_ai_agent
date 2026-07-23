"""
model.py
Contains Hugging Face intent classification logic
and Gemini API integrations with rate-limit/quota fallback handling.
"""

import os
import re
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

# ---------------------------------------------------------------------------
# 1. Hugging Face Intent Model Setup & Fine-Tuning Pipeline
# ---------------------------------------------------------------------------

MODEL_NAME = "distilbert-base-uncased"

def fine_tune_hf_classifier(dataset_dict, label2id: Dict[str, int]):
    """Fine-tunes DistilBERT on field service ticket data."""
    id2label = {v: k for k, v in label2id.items()}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_func(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    tokenized_datasets = dataset_dict.map(tokenize_func, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="./fine_tuned_intent_model",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        use_cpu=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        processing_class=tokenizer,
    )

    print("[HF Model] Starting fine-tuning...")
    trainer.train()
    model.save_pretrained("./fine_tuned_intent_model")
    tokenizer.save_pretrained("./fine_tuned_intent_model")
    print("[HF Model] Model fine-tuning complete.")

# ---------------------------------------------------------------------------
# 2. Pydantic Schema & Mock Fallback Extractor
# ---------------------------------------------------------------------------

class ServiceTicketExtraction(BaseModel):
    """Pydantic schema for structured output extraction."""
    service_type: str = Field(description="Type of field service required")
    urgency: str = Field(description="Urgency level: LOW, MEDIUM, HIGH, or CRITICAL")
    location: str = Field(description="Client location or branch mentioned in the ticket")
    required_skills: List[str] = Field(description="List of technical skills needed")
    summary: str = Field(description="One-sentence summary of the customer's problem")

def _rule_based_fallback_extraction(email_text: str) -> ServiceTicketExtraction:
    """Fallback parser used when API credits/quotas are exhausted."""
    text_lower = email_text.lower()
    
    # Urgency heuristic
    urgency = "LOW"
    if any(w in text_lower for w in ["asap", "urgent", "sparks", "risk", "immediately", "emergency"]):
        urgency = "HIGH"
    elif any(w in text_lower for w in ["stopped", "broken", "leak", "tripped"]):
        urgency = "MEDIUM"

    # Service & skill heuristic
    service_type = "General Maintenance"
    skills = ["General Inspection"]
    if "freezer" in text_lower or "refrigeration" in text_lower or "cooling" in text_lower:
        service_type = "Commercial Refrigeration"
        skills = ["Refrigeration", "Chillers"]
    elif "hvac" in text_lower or "ac" in text_lower or "air" in text_lower:
        service_type = "HVAC Repair"
        skills = ["HVAC", "Compressors"]
    elif "breaker" in text_lower or "sparks" in text_lower or "electrical" in text_lower:
        service_type = "Electrical Repair"
        skills = ["High Voltage", "Electrical"]

    # Location heuristic
    location = "Main Facility"
    if "downtown" in text_lower:
        location = "Downtown Branch"
    elif "bay" in text_lower or "industrial" in text_lower:
        location = "Industrial Bay"

    return ServiceTicketExtraction(
        service_type=service_type,
        urgency=urgency,
        location=location,
        required_skills=skills,
        summary=f"Automated offline fallback: Request regarding {service_type.lower()}."
    )

# ---------------------------------------------------------------------------
# 3. LangChain Extractor Runnable Wrapper
# ---------------------------------------------------------------------------

class ResilientExtractorRunnable:
    """Runnable that tries Gemini API models and falls back to rule-based logic on 429 quota errors."""
    def __init__(self, api_key: str):
        self.api_key = api_key.strip()

    def invoke(self, input_dict: Dict[str, Any]) -> ServiceTicketExtraction:
        email_text = input_dict.get("email_text", "")
        os.environ["GOOGLE_API_KEY"] = self.api_key
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert dispatcher for a field service company. "
                       "Analyze the incoming customer request and extract operational entities accurately."),
            ("human", "{email_text}")
        ])

        # Attempt Gemini API call
        for model_name in MODEL_CANDIDATES:
            try:
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    api_key=self.api_key,
                    temperature=0.1
                )
                chain = prompt | llm.with_structured_output(ServiceTicketExtraction)
                return chain.invoke({"email_text": email_text})
            except Exception as e:
                err_str = str(e)
                if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                    print("[Model Warning] Quota exhausted (429). Using offline fallback parser.")
                    return _rule_based_fallback_extraction(email_text)
                continue
                
        # If all model candidate attempts failed for other reasons, use fallback
        return _rule_based_fallback_extraction(email_text)

def get_gemini_extractor(api_key: str):
    """Returns a resilient extractor object that matches LangChain's .invoke() interface."""
    return ResilientExtractorRunnable(api_key)

def draft_customer_response(api_key: str, extracted_info: Dict[str, Any], tech_name: str, slot: str) -> str:
    """Generates a customer response email using Gemini or an offline template fallback."""
    cleaned_key = api_key.strip()
    os.environ["GOOGLE_API_KEY"] = cleaned_key

    for model_name in MODEL_CANDIDATES:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                api_key=cleaned_key,
                temperature=0.3
            )
            prompt = f"""
            Write a polite, professional response email to a field service customer based on these details:
            - Service Type: {extracted_info.get('service_type')}
            - Priority: {extracted_info.get('urgency')}
            - Scheduled Technician: {tech_name}
            - Estimated Arrival: {slot}
            
            Keep the email concise, warm, and reassuring.
            """
            response = llm.invoke(prompt)
            return response.content
        except Exception:
            continue

    # Offline template fallback if API fails
    return (
        f"Hello,\n\nThank you for reaching out to Field Service Operations. "
        f"We have registered your {extracted_info.get('urgency', 'NORMAL')} priority request "
        f"for {extracted_info.get('service_type', 'service')}. "
        f"Technician {tech_name} has been assigned and is scheduled for {slot}.\n\n"
        f"Best regards,\nDispatch Team"
    )