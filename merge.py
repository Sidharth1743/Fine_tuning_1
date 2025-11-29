from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# 1. Setup Paths (Use Absolute Paths to be safe)
base_model_id = "meta-llama/Llama-3.2-1B"
adapter_path = "/home/sidharth/Desktop/Finetune/local_adapter"
output_path = "/home/sidharth/Desktop/Finetune/merged_model"

print(f"Loading base model: {base_model_id}")
# Load base model in float16 on CPU (fits easily in RAM)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    return_dict=True,
    dtype=torch.bfloat16,
    device_map="auto",
    token=os.getenv("HF_TOKEN", "")
)

print(f"Loading adapter from: {adapter_path}")
model = PeftModel.from_pretrained(base_model, adapter_path)

print("Merging adapter into base model...")
model = model.merge_and_unload()

print(f"Saving merged model to: {output_path}")
model.save_pretrained(output_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.save_pretrained(output_path)

print("Done! You can now use this in Ollama.")
