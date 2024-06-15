import json
import random
from dotenv import load_dotenv
load_dotenv(".env")
import time
from openai import OpenAI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



client = OpenAI()
def call_gpt(messages, model_name="gpt-4o"):
    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=1,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
                )
            time.sleep(1)
            break
        except Exception as e:
            print(e)
            time.sleep(30)
    return response.choices[0].message['content']

def human_icl(num_shot):
    ##############################################
    # generate human icl with MQuAKE dataset
    ##############################################

    with open('datasets/MQuAKE-CF-3k.json', 'r') as f:
        data = json.load(f)
    human_icl = []
    sampled = random.sample(data, num_shot)
    for entry in sampled:
        fact_prefix = "New Fact: "
        new_facts = []
        for rewrite in entry["requested_rewrite"]:
            new_facts.append(rewrite["prompt"].format(rewrite["subject"]) + " " + rewrite["target_new"]["str"])
        
        thoughts = []
        for thought in entry["single_hops"]:
            thoughts.append(thought["cloze"] + " " + thought["answer"] + ".")
        thoughts = " ".join(thoughts)
        human_icl.append(fact_prefix + ", ".join(new_facts) + "\n" + "Question: " + entry['questions'][0] + "\n" + "Thought: " + thoughts + "\n" + "Answer: " + entry["new_answer"])
    return "\n\n".join(human_icl)

def chatgpt_icl_by_human_icl(model_name, num_shot, num_human_icl):
    ##############################################
    # generate human icl
    ##############################################
    with open('datasets/MQuAKE-CF-3k.json', 'r') as f:
        data = json.load(f)
    human_icl = []
    sampled = random.sample(data, num_human_icl)
    for entry in sampled:
        fact_prefix = "New Fact: "
        new_facts = []
        for rewrite in entry["requested_rewrite"]:
            new_facts.append(rewrite["prompt"].format(rewrite["subject"]) + " " + rewrite["target_new"]["str"])
        
        thoughts = []
        for thought in entry["single_hops"]:
            thoughts.append(thought["cloze"] + " " + thought["answer"] + ".")
        thoughts = " ".join(thoughts)
        human_icl.append(fact_prefix + ", ".join(new_facts) + "\n" + "Question: " + entry['questions'][0] + "\n" + "Thought: " + thoughts + "\n" + "Answer: " + entry["new_answer"])
    human_icl = "\n\n".join(human_icl)
    
    chatgpt_prompt = \
f'''
Your task is to genereate knowledge editing examples for in context learning.
You need to first generate the knowledge being edited (fact being changed) and then ask a question that requires multi-hop (multi-step) reasoning. Finally you need to provide a answer with step-by-step reasoning in concise format.

Example:
{human_icl}

Please respond in the following format without any markdown.
New Fact: <knowledge being editted>
Question: <question that requires multi-step reasoning>
Thought: <step-by-step reasoning in concise format>
Answer: <answer with step-by-step reasoning in concise format>

Please generate {num_shot} knowledge editing examples. Please respond only the generated examples in the above format without any markdown or additional text.
'''
    messages = [{
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": "You are a helpful AI assistant."
            }
        ]
        }, {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": chatgpt_prompt
                }
            ]
        }
    ]
    response = call_gpt(messages, model_name)
    return response


def chatgpt_icl_by_zeroshot(model_name, num_shot):
    chatgpt_prompt = \
f'''
Your task is to genereate knowledge editing examples for in context learning.
You need to first generate the knowledge being edited (fact being changed) and then ask a question that requires multi-hop (multi-step) reasoning. Finally you need to provide a answer with step-by-step reasoning in concise format.

Please respond in the following format without any markdown.
New Fact: <knowledge being editted>
Question: <question that requires multi-step reasoning>
Thought: <step-by-step reasoning in concise format>
Answer: <answer with step-by-step reasoning in concise format>

Please generate {num_shot} knowledge editing examples. Please respond only the generated examples in the above format without any markdown or additional text.
'''
    messages = [{
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": "You are a helpful AI assistant."
            }
        ]
        }, {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": chatgpt_prompt
                }
            ]
        }
    ]
    response = call_gpt(messages, model_name)
    return response

def gptj_icl_by_human_icl(num_human_icl, num_shot):
    ##############################################
    # generate human icl
    ##############################################
    with open('datasets/MQuAKE-CF-3k.json', 'r') as f:
        data = json.load(f)
    human_icl = []
    sampled = random.sample(data, num_human_icl)
    for entry in sampled:
        fact_prefix = "New Fact: "
        new_facts = []
        for rewrite in entry["requested_rewrite"]:
            new_facts.append(rewrite["prompt"].format(rewrite["subject"]) + " " + rewrite["target_new"]["str"])
        
        thoughts = []
        for thought in entry["single_hops"]:
            thoughts.append(thought["cloze"] + " " + thought["answer"] + ".")
        thoughts = " ".join(thoughts)
        human_icl.append(fact_prefix + ", ".join(new_facts) + "\n" + "Question: " + entry['questions'][0] + "\n" + "Thought: " + thoughts + "\n" + "Answer: " + entry["new_answer"])
    human_icl = "\n\n".join(human_icl)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").to(device)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    
    prompt = \
f'''
Your task is to genereate knowledge editing examples for in context learning.
You need to first generate the knowledge being edited (fact being changed) and then ask a question that requires multi-hop (multi-step) reasoning. Finally you need to provide a answer with step-by-step reasoning in concise format.

Example:
{human_icl}

Please generate {num_shot} knowledge editing examples. Please respond only the generated examples in the above format without any markdown or additional text.
'''
    print(prompt)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=1.0,
        max_length=4096,
        max_new_tokens=50,
    )
    generated_token_ids = gen_tokens[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    print(generated_text)
    
if __name__ == "__main__":
    gptj_icl_by_human_icl(num_human_icl=3, num_shot=5)
