import json
import torch
import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

human_icl = '''
New fact: the author of Misery is Richard Dawkins.
Question: What is the nationality of the author of Misery.
Thought: The author of Misery is Richard Dawkins. Richard Dawkins is a citizen of United Kingdom. Therefore, the nationality of the author of Misery is British.
Answer: British

New fact: The capital of United States of America is El Campu.
Question: What is the capital city of the country that Michael Feinstein is a citizen of?
Thought: Michael Feinstein is a citizen of United States of America. The capital of United States of America is El Campu. Thus, the capital city of the country that Michael Feinstein is a citizen of is El Campu.
Answer: El Campu
'''


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
    
        prompt = human_icl + "\n\n" + fact_prefix + ", ".join(new_facts) + "\n" + "Question: " + question + "\n"
        
        
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
