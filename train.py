from datasets import load_dataset
from colorama import Fore
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM , BitsAndBytesConfig
from trl import SFTTrainer , SFTConfig
from peft import LoraConfig,prepare_model_for_kbit_training
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

dataset = load_dataset("data",split='train')
print(Fore.YELLOW + str(dataset[10]) + Fore.RESET)

def format_chat_template(batch , tokenizer):
    system_prompt = """You are a helpful, honest and harmless assitant designed to help engineers. Think through each question logically and provide an answer. Don't make things up, if you're unable to answer a question advise the user that you're unable to answer as it is outside of your scope."""
    samples =[]
    questions = batch["question"]
    answers = batch["answer"]
    for i in range(len(questions)):
        row_json = [
            {"role" : "system" , "content":system_prompt},
            {"role": "user" , "content":questions[i]},
            {"role": "assistant" , "content":answers[i]}
        ]

        tokenizer.chat_template = """
        {% set loop_messages = messages %}
        {% for message in loop_messages %}
        {% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}
        {% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}
        {{ content }}
        {% endfor %}
        {% if add_generation_prompt %}
        {{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
        {% endif %}
        """

        text = tokenizer.apply_chat_template(row_json, tokenize=False)
        samples.append(text)

    return {
        "instruction":questions,
        "response": answers,
        "text":samples
    }

base_model = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    trust_remote_code=True,
    token = os.getenv("HF_TOKEN", "")
)

train_dataset = dataset.map(lambda x:format_chat_template(x,tokenizer), num_proc=8 , batched = True , batch_size = 10)

print(Fore.GREEN + str(train_dataset[10]) + Fore.RESET)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True , 
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    token=os.getenv("HF_TOKEN", ""),
    cache_dir="/home/sidharth/Desktop/Finetune/cache",
    quantization_config=bnb_config
)

print(Fore.CYAN + str(model) + Fore.RESET)
print(Fore.YELLOW + str(next(model.parameters()).device)+ Fore.RESET)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=256,
    lora_alpha=512,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM"
)

training_args = SFTConfig(
    output_dir="meta-llama/llama-3.2-1B-SFT_1",
    num_train_epochs=50,                     # Start with 1 to test
    per_device_train_batch_size=1,          # CRITICAL: Must be 1 for 4GB VRAM
    gradient_accumulation_steps=4,          # Accumulate to simulate larger batch (effective batch = 4)
    logging_steps=10,
    save_steps=150,
    max_length=512,                     # CRITICAL: Reduce context length. 1024 or 2048 will OOM.
    packing=False,                          # Don't pack sequences
    optim="paged_adamw_8bit",               # CRITICAL: Saves GPU memory by using RAM
    bf16=True,                              # Use mixed precision (or bf16 if your GPU supports it)
)
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    peft_config=peft_config,
)

trainer.train()

trainer.save_model('complete_checkpoint')
trainer.model.save_pretrained("final_model")