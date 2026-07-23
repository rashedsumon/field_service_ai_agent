"""
data_loader.py
Handles downloading, generating, and preparing field service datasets 
from Hugging Face and local sources for fine-tuning and evaluation.
"""

import os
import pandas as pd
from datasets import Dataset, DatasetDict

def generate_mock_field_service_dataset() -> pd.DataFrame:
    """Generates a structured dataset of customer field service requests for intent classification."""
    data = [
        {
            "text": "Our rooftop AC unit is leaking water into the main office hallway and making a grinding noise.",
            "intent": "HVAC Repair",
            "urgency": "HIGH",
            "skills": "HVAC, Compressors"
        },
        {
            "text": "Need a routine annual inspection for our warehouse fire alarm and sprinkler systems.",
            "intent": "Maintenance",
            "urgency": "LOW",
            "skills": "Fire Safety, Inspection"
        },
        {
            "text": "Main breaker keeps tripping whenever we turn on the heavy machinery in Bay 3.",
            "intent": "Electrical Repair",
            "urgency": "HIGH",
            "skills": "High Voltage, Commercial Electrical"
        },
        {
            "text": "Can someone quote us for adding two new ethernet drop points in the conference room?",
            "intent": "Installation",
            "urgency": "MEDIUM",
            "skills": "Low Voltage, Cabling"
        },
        {
            "text": "The walk-in freezer temperature is rising. It is currently at 45 degrees and climbing.",
            "intent": "Commercial Refrigeration",
            "urgency": "CRITICAL",
            "skills": "Refrigeration, Chillers"
        }
    ]
    return pd.DataFrame(data)

def load_and_prepare_hf_dataset(save_dir: str = "./data") -> DatasetDict:
    """
    Downloads or builds the dataset, formats it into a Hugging Face DatasetDict, 
    and saves it to disk.
    """
    os.makedirs(save_dir, exist_ok=True)
    df = generate_mock_field_service_dataset()
    
    # Map intent strings to numerical labels
    unique_intents = sorted(df["intent"].unique().tolist())
    intent2id = {intent: idx for idx, intent in enumerate(unique_intents)}
    df["label"] = df["intent"].map(intent2id)
    
    # Convert to HF Dataset
    hf_dataset = Dataset.from_pandas(df)
    
    # Train / Test Split
    ds_dict = hf_dataset.train_test_split(test_size=0.2, seed=42)
    
    csv_path = os.path.join(save_dir, "field_service_tickets.csv")
    df.to_csv(csv_path, index=False)
    print(f"[DataLoader] Dataset saved successfully to {csv_path}")
    
    return ds_dict, intent2id

if __name__ == "__main__":
    dataset, labels = load_and_prepare_hf_dataset()
    print("[DataLoader] Sample split:", dataset)
    print("[DataLoader] Label mappings:", labels)