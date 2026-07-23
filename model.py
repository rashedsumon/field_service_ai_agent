"""
model.py
Contains Hugging Face intent classification logic
and Gemini API integrations via LangChain with automatic model fallbacks.
"""

import os
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Active Gemini model candidates in order of preference
MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-1.5-pro",
    "gemini-2.5-pro",
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
# 2. LangChain + Google Gemini Structured Extractor
# ---------------------------------------------------------------------------

class ServiceTicketExtraction(BaseModel):
    """Pydantic schema for structured output extraction using Gemini."""
    service_type: str = Field(description="Type of field service required (e.g., HVAC, Electrical, Refrigeration)")
    urgency: str = Field(description="Urgency level: LOW, MEDIUM, HIGH, or CRITICAL")
    location: str = Field(description="Client location or branch mentioned in the ticket")
    required_skills: List[str] = Field(description="List of technical skills needed to resolve the issue")
    summary: str = Field(description="One-sentence summary of the customer's problem")

def _initialize_llm_with_fallback(api_key: str, temperature: float = 0.1):
    """
    Attempts to initialize ChatGoogleGenerativeAI across model candidates 
    to handle deprecated or region-locked endpoints gracefully.
    """
    cleaned_key = api_key.strip()
    os.environ["GOOGLE_API_KEY"] = cleaned_key
    
    last_error = None
    for model_name in MODEL_CANDIDATES:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                api_key=cleaned_key,
                temperature=temperature
            )
            return llm, model_name
        except Exception as e:
            last_error = e
            continue
            
    raise RuntimeError(f"Failed to initialize any Gemini model. Last error: {last_error}")

def get_gemini_extractor(api_key: str):
    """Initializes Gemini model configured with structured output parsing."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert dispatcher for a field service company. "
                   "Analyze the incoming customer request and extract operational entities accurately."),
        ("human", "{email_text}")
    ])

    last_error = None
    cleaned_key = api_key.strip()
    os.environ["GOOGLE_API_KEY"] = cleaned_key

    # Try each candidate model until structured output succeeds
    for model_name in MODEL_CANDIDATES:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                api_key=cleaned_key,
                temperature=0.1
            )
            structured_llm = llm.with_structured_output(ServiceTicketExtraction)
            return prompt | structured_llm
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"All candidate models failed during extraction setup. Last error: {last_error}")

def draft_customer_response(api_key: str, extracted_info: Dict[str, Any], tech_name: str, slot: str) -> str:
    """Generates a professional customer response email using Gemini."""
    llm, _ = _initialize_llm_with_fallback(api_key, temperature=0.3)
    
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