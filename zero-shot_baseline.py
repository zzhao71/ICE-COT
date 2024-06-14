import json
import torch
import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

with open('datasets/MQuAKE-CF-3k.json', 'r') as f:
    dataset = json.load(f)
    
def is_answer_correct(provided_answer, correct_answers):
    return any(correct_answer.lower() in provided_answer.lower() for correct_answer in correct_answers)

correct_answers = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").to(device)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")


use_imagine = False

total = 0
correct = 0

for i, entry in enumerate(tqdm.tqdm(dataset)):
    fact_prefix = "New Fact: " if not use_imagine else "Imagine "
    new_facts = []
    for rewrite in entry["requested_rewrite"]:
        new_facts.append(rewrite["prompt"].format(rewrite["subject"]) + " " + rewrite["target_new"]["str"])
    
    for question in entry['questions']:
    
        prompt = fact_prefix + ", ".join(new_facts) + "\n" + "Question: " + question + "\n"
        
        
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

        answers = entry['new_answer_alias']
        answers.append(entry["new_answer"])
        if is_answer_correct(generated_text, answers):
            correct += 1
            break
    total += 1
    
print(f"Correct: {correct}/{total} ({correct/total*100:.2f}%)")
