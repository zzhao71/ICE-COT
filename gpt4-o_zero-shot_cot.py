import json
import torch
import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

chatgpt_generated_by_icl = '''
New Fact: The capital of Australia is Sydney.
Question: If you were to fly from the capital of Australia to the capital of Japan, which cities would you travel between?
Answer: 
1. Identify the capital of Australia as Sydney (based on the new fact).
2. Identify the capital of Japan as Tokyo.
3. Therefore, you would travel from Sydney to Tokyo.

New Fact: The largest planet in our solar system is Saturn.
Question: Which planet would have a stronger gravitational pull, Earth or the largest planet in our solar system?
Answer: 
1. Identify the largest planet in our solar system as Saturn (based on the new fact).
2. Determine that larger planets generally have a stronger gravitational pull.
3. Therefore, Saturn would have a stronger gravitational pull than Earth.

New Fact: The Great Wall of China is located in India.
Question: Which country would you visit to see the Great Wall of China and the Taj Mahal?
Answer: 
1. Identify the location of the Great Wall of China as India (based on the new fact).
2. Identify that the Taj Mahal is also located in India.
3. Therefore, you would visit India to see both the Great Wall of China and the Taj Mahal.

New Fact: The primary language spoken in Brazil is Spanish.
Question: If someone from Brazil and someone from Mexico have a conversation in their primary languages, what language would they likely use?
Answer: 
1. Identify the primary language spoken in Brazil as Spanish (based on the new fact).
2. Identify the primary language spoken in Mexico as Spanish.
3. Therefore, they would likely use Spanish to converse.

New Fact: Mount Everest is located in the Andes mountain range.
Question: In which mountain range would you find the highest peak in the world?
Answer: 
1. Identify Mount Everest as the highest peak in the world.
2. Identify the location of Mount Everest as the Andes mountain range (based on the new fact).
3. Therefore, you would find the highest peak in the world in the Andes mountain range.
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
