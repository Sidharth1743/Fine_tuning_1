from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from colorama import Fore
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions

from typing import List
import json
from pydantic import BaseModel
from litellm import completion
from generated_prompt import prompt_template
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set environment variables from .env file
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY_SYNTHETIC", "")

# Configure accelerator options for GPU
accelerator_options = AcceleratorOptions(
    device=AcceleratorDevice.CUDA,  # or AcceleratorDevice.AUTO
)

class Record(BaseModel):
    question : str
    answer  :str

class Response(BaseModel):
    generated : List[Record]

def llm_call(data:str , num_records : int = 5) -> dict:
    stream = completion(
        model = "gemini/gemini-2.0-flash",
        messages = [
            {
                "role" : "user",
                "content": prompt_template(data, num_records),
            }
        ],
        stream = True,
        max_tokens = 2000, # Changed from num_predict to max_tokens for better compatibility
        response_format = Response # Ensure you pass the Pydantic class here if supported by the version
    )

    generated_text = ""
    for x in stream:
        # valid check for delta content
        if x.choices and x.choices[0].delta.content:
            delta = x.choices[0].delta.content
            print(Fore.YELLOW + delta + Fore.RESET , end ="")
            generated_text += delta
    
    # --- FIX STARTS HERE ---
    # Remove markdown code fences if they exist
    cleaned_text = generated_text.replace("```json", "").replace("```", "").strip()
    
    return json.loads(cleaned_text)


if __name__ == "__main__":
    converter =  DocumentConverter()
    doc = converter.convert("/home/sidharth/Desktop/Finetune/DRL.pdf").document
    chunker = HybridChunker()
    chunks = chunker.chunk(dl_doc=doc)

    dataset = {}
    for i , chunk in enumerate(chunks):
        print(Fore.CYAN + f"Raw Text:/n {chunk.text[:300]}.."+ Fore.RESET)
        enriched_text = chunker.contextualize(chunk=chunk)
        print(Fore.YELLOW+ f"Contextualized Text: \n{enriched_text[:300]}" + Fore.RESET)
        data = llm_call(enriched_text)
        dataset[i] = {
            "generated":data['generated'],
            "context":enriched_text
                      }
    with open('DRL.json','w') as f:
        json.dump(dataset,f)
