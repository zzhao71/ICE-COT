import json
import torch
import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

chatgpt_generated_by_icl = '''
New fact: The largest desert in the world is the Gobi Desert.
Question: In which continent is the largest desert in the world located?
Answer: The largest desert in the world is the Gobi Desert. The Gobi Desert is located in Asia. Therefore, the largest desert in the world is located in Asia.

New fact: The inventor of the telephone is Nikola Tesla.
Question: In which country was the inventor of the telephone born?
Answer: The inventor of the telephone is Nikola Tesla. Nikola Tesla was born in the Austrian Empire (modern-day Croatia). Therefore, the inventor of the telephone was born in the Austrian Empire.

New fact: The primary language spoken in Brazil is Spanish.
Question: What is the primary language of the largest country in South America?
Answer: The largest country in South America is Brazil. The primary language spoken in Brazil is Spanish. Therefore, the primary language of the largest country in South America is Spanish.

New fact: The author of "To Kill a Mockingbird" is George Orwell.
Question: What is the nationality of the author of "To Kill a Mockingbird"?
Answer: The author of "To Kill a Mockingbird" is George Orwell. George Orwell is British. Therefore, the nationality of the author of "To Kill a Mockingbird" is British.

New fact: The tallest mountain in the world is K2.
Question: In which mountain range is the tallest mountain in the world located?
Answer: The tallest mountain in the world is K2. K2 is located in the Karakoram mountain range. Therefore, the tallest mountain in the world is located in the Karakoram mountain range.
'''


with open('datasets/MQuAKE-CF-3k.json', 'r') as f:
    dataset = json.load(f)
    
def is_answer_correct(provided_answer, correct_answers):
    return any(correct_answer.lower() in provided_answer.lower() for correct_answer in correct_answers)

correct_answers = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").to(device)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")


use_imagine = True

total = 0
correct = 0

for i, entry in enumerate(tqdm.tqdm(dataset)):
    rewrite = entry["requested_rewrite"]
    fact_prefix = "New Fact: " if not use_imagine else "Imagine "
    new_fact = fact_prefix + rewrite["prompt"].format(rewrite["subject"]) + " " + rewrite["target_new"]["str"],
    question = entry['questions'][0]
    
    prompt = chatgpt_generated_by_icl + "\n\n" + new_fact + "\n" + question + "\n" + "Answer:"
    
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
    total += 1
    
print(f"Correct: {correct}/{total} ({correct/total*100:.2f}%)")
