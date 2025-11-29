import json
from colorama import Fore

instructions=[]
with open('DRL.json','r') as f:
    data = json.load(f)
    for key ,chunk in data.items():
        for pairs in chunk['generated']:
            question,answer = pairs['question'],pairs['answer']
            context_pair = {
                'question'  : f'{pairs['question']}',
                'answer' : pairs['answer']
            }
            instructions.append(context_pair)
        print(Fore.YELLOW + str(chunk))

with open('data/instruction_format.json','w') as f:
    json.dump(instructions, f)
