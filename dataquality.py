import json
from pydantic import BaseModel
from litellm import completion
from colorama import Fore
import os
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set environment variables from .env file
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY_DATAQUALITY", "")
os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY", "")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY", "")
class Score(BaseModel):
    score : int
    explanation : str

class Rank(BaseModel):
    accuracy : Score
    style : Score

def llm_call(record: dict):
    prompt = f"""
Evaluate the given instruction-tuning record on two dimensions: accuracy and style.  
Each score must be an integer from 1 to 10.

Scoring Rules:

Accuracy:
- If the question is not factual, assign accuracy = 0.
- If the answer fails to properly answer the question, assign accuracy = 1.
- Otherwise, assign a score from 2–10 based on how well the answer satisfies the question.

Style:
- If the question or answer contains anything harmful, unhelpful, or dishonest, assign style = 1.
- If the question or answer is blank or consists only of symbols such as "...", assign both accuracy and style = 1.
- Otherwise, assign a score from 2–10 based on clarity, coherence, and overall quality.

You must also provide a brief explanation for why you assigned each score.  
Your output must be fully self-contained and must not rely on external context.

Input Record:
{record}

Return strictly JSON:
{{
    "accuracy": {{
        "score": 0,
        "explanation": ""
    }},
    "style": {{
        "score": 0,
        "explanation": ""
    }}
}}
"""

    # Retry parameters
    max_retries = 5
    backoff = 2

    for attempt in range(max_retries):
        try:
            stream = completion(
                model="groq/moonshotai/kimi-k2-instruct",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                max_tokens=2000,
            )

            data = ""
            for x in stream:
                delta = x["choices"][0]["delta"]["content"]
                if delta:
                    print(Fore.YELLOW + delta + Fore.RESET, end="")
                    data += delta

            cleaned = data.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)

        except Exception as e:
            if "rate_limit" in str(e).lower() or "limit" in str(e).lower():
                wait = backoff ** attempt
                print(Fore.RED + f"\nRate limit hit. Retrying in {wait} seconds..." + Fore.RESET)
                time.sleep(wait)
                continue
            else:
                raise e

    raise RuntimeError("Max retries exceeded during LLM call.")

if __name__ == "__main__":
    quality = []
    instructions=[]
    with open('data/instruction_format.json' , 'r') as f:
        data = json.load(f)
        for pair in data:
            print(Fore.YELLOW + str(pair) + Fore.RESET)
            result = llm_call(pair)

            if result['accuracy']['score'] >= 6 and result['style']['score'] >= 6:
                instructions.append(pair)
                quality.append({**pair , 'quality' : result})

    with open('data/insturctionquality.json' , 'w') as f:
        json.dump(instructions,f)

    with open('qualityresults.json','w') as f:
        json.dump(quality, f)