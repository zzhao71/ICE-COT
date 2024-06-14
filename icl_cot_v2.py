import json
import torch
import tqdm
from icl_prompt import human_icl, chatgpt_generated_by_icl, chatgpt_generated_by_zeroshot
from transformers import AutoModelForCausalLM, AutoTokenizer

import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--type", type=str)
argparser.add_argument("--json", type=str)
argparser.add_argument("--out_dir", type=str)

args = argparser.parse_args()

if args.type == "chatgpt_icl":
    icl_prompt = chatgpt_generated_by_icl + "\n\n"
elif args.type == "chatgpt_zeroshot":
    icl_prompt = chatgpt_generated_by_zeroshot + "\n\n"
elif args.type == "human":
    icl_prompt = human_icl + "\n\n"
elif args.type == "none":
    icl_prompt = ""


with open(args.json, 'r') as f:
    dataset = json.load(f)
    
def is_answer_correct(provided_answer, correct_answers):
    return any(correct_answer.lower() in provided_answer.lower() for correct_answer in correct_answers)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").to(device)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

output = []
for i, entry in enumerate(tqdm.tqdm(dataset)):
    
    answers = []
    fact_prefix = "New Fact: "
    new_facts = []
    for rewrite in entry["requested_rewrite"]:
        new_facts.append(rewrite["prompt"].format(rewrite["subject"]) + " " + rewrite["target_new"]["str"])
    
    for question in entry['questions']:
    
        prompt = icl_prompt + fact_prefix + ", ".join(new_facts) + "\n" + "Question: " + question + "\n"
        
        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        gen_tokens = model.generate(
            input_ids,
            do_sample=True,
            temperature=0.9,
            max_length=500,
            max_new_tokens=50,
        )
        generated_token_ids = gen_tokens[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        answers.append(generated_text)
    
    entry['output_by_model'] = answers
    output.append(entry)


with open(args.out_dir, 'w') as f:
    json.dump(output, f)
    
