import json
import random
from dotenv import load_dotenv
load_dotenv(".env")
import time
from openai import OpenAI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset import load_ripple


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

    return response.choices[0].message.content
def human_icl(json_file, num_shot):
    questions = load_ripple(json_file)
    sampled = random.sample(questions, num_shot)
    output = []
    for entry in sampled:
        output.append(entry["formated_questions"][0] + "Answer: " + entry["formated_answers"][-1] + "\n")
    return "\n\n".join(output)


def human_icl_cot(num_shot):
    ##############################################
    # generate human icl with MQuAKE dataset
    ##############################################

    selected_human_icl_cot = [
        """New fact: the author of Misery is Richard Dawkins.
Prompt: The nationality of the author of Misery is 
Thought: The author of Misery is Richard Dawkins. Richard Dawkins is a citizen of United Kingdom. Therefore, the nationality of the author of Misery is British.
Answer: British""",
        """New fact: The capital of United States of America is El Campu.
Prompt: The capital city of Michael Feinstein's nationality is
Thought: Michael Feinstein is a citizen of United States of America. The capital of United States of America is El Campu. Thus, the capital city of the country that Michael Feinstein is a citizen of is El Campu.
Answer: El Campu""",
        """New fact: William Gillette is a citizen of Indonesia, The official language of Indonesia is German.
Prompt: The official language of William Gillette's country of citizenship is
Thought: William Gillette is a citizen of Indonesia. The official language of Indonesia is German. Therefore, the official language of William Gillette's country of citizenship is German.
Answer: German
""",
        """New fact: Chauncey Billups is associated with the sport of pes\u00e4pallo.
Prompt: The current leader of the country from which Chauncey Billups originates is
Thought: Chauncey Billups is associated with the sport of pes\u00e4pallo. Pes\u00e4pallo was created in the country of Finland. The name of the current head of state in Finland is Sauli Niinist\u00f6. Therefore, the current leader of the country from which Chauncey Billups originates is Sauli Niinist\u00f6.
Answer: Sauli Niinist\u00f6
""",
        """New fact: Ellie Kemper is a citizen of Croatia.
Prompt: The head of state of the country where Ellie Kemper holds a citizenship is
Thought: Ellie Kemper is a citizen of Croatia. The name of the current head of state in Croatia is Kolinda Grabar-Kitarovi\u0107. Therefore, the head of state of the country where Ellie Kemper holds a citizenship is Kolinda Grabar-Kitarovi\u0107.
Answer: Kolinda Grabar-Kitarovi\u0107
"""
    ]
    
    return "\n\n".join(selected_human_icl_cot[:num_shot])

def chatgpt_icl_cot_by_human_icl_cot(model_name, num_shot, num_human_icl):
    ##############################################
    # generate human icl
    ##############################################
    selected_human_icl_cot = [
        """New fact: the author of Misery is Richard Dawkins.
Prompt: The nationality of the author of Misery is 
Thought: The author of Misery is Richard Dawkins. Richard Dawkins is a citizen of United Kingdom. Therefore, the nationality of the author of Misery is British.
Answer: British""",
        """New fact: The capital of United States of America is El Campu.
Prompt: The capital city of Michael Feinstein's nationality is
Thought: Michael Feinstein is a citizen of United States of America. The capital of United States of America is El Campu. Thus, the capital city of the country that Michael Feinstein is a citizen of is El Campu.
Answer: El Campu""",
        """New fact: William Gillette is a citizen of Indonesia, The official language of Indonesia is German.
Prompt: The official language of William Gillette's country of citizenship is
Thought: William Gillette is a citizen of Indonesia. The official language of Indonesia is German. Therefore, the official language of William Gillette's country of citizenship is German.
Answer: German
""",
        """New fact: Chauncey Billups is associated with the sport of pes\u00e4pallo.
Prompt: The current leader of the country from which Chauncey Billups originates is
Thought: Chauncey Billups is associated with the sport of pes\u00e4pallo. Pes\u00e4pallo was created in the country of Finland. The name of the current head of state in Finland is Sauli Niinist\u00f6. Therefore, the current leader of the country from which Chauncey Billups originates is Sauli Niinist\u00f6.
Answer: Sauli Niinist\u00f6
""",
        """New fact: Ellie Kemper is a citizen of Croatia.
Prompt: The head of state of the country where Ellie Kemper holds a citizenship is
Thought: Ellie Kemper is a citizen of Croatia. The name of the current head of state in Croatia is Kolinda Grabar-Kitarovi\u0107. Therefore, the head of state of the country where Ellie Kemper holds a citizenship is Kolinda Grabar-Kitarovi\u0107.
Answer: Kolinda Grabar-Kitarovi\u0107
"""
    ]
    
    
    human_icl = "\n\n".join(selected_human_icl_cot[:num_human_icl])
    
    chatgpt_prompt = \
f'''
Your task is to genereate knowledge editing examples for in context learning.
You need to first generate the knowledge being edited (fact being changed) and then ask a question that requires multi-hop (multi-step) reasoning. Finally you need to provide a answer with step-by-step reasoning in concise format.

Example:
{human_icl}

Please respond in the following format without any markdown.
New Fact: <knowledge being editted>
Prompt: <question that requires multi-step reasoning in the form of a prompt that lead to the answer right after the prompt>
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


def chatgpt_icl_cot_by_zeroshot(model_name, num_shot):
    chatgpt_prompt = \
f'''
Your task is to genereate knowledge editing examples for in context learning.
You need to first generate the knowledge being edited (fact being changed) and then ask a question that requires multi-hop (multi-step) reasoning. Finally you need to provide a answer with step-by-step reasoning in concise format.

Please respond in the following format without any markdown.
New Fact: <knowledge being editted>
Prompt: <question that requires multi-step reasoning in the form of a prompt that lead to the answer right after the prompt>
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

def gptj_icl_cot_by_human_icl_cot(num_human_icl, num_shot, generate_on_the_fly=True):
    if generate_on_the_fly:
        ##############################################
        # generate human icl
        ##############################################
        with open('datasets/mquake/MQuAKE-CF-3k.json', 'r') as f:
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
            human_icl.append(fact_prefix + ", ".join(new_facts) + "\n" + "Prompt: " + entry['questions'][0] + "\n" + "Thought: " + thoughts + "\n" + "Answer: " + entry["new_answer"])
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").to(device)
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        
        human_icl = "\n\n".join(human_icl)
        
        generated_icl = []
        for i in range(num_shot):
            prompt = human_icl + "\n\nNew Fact: "
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

            gen_tokens = model.generate(
                input_ids,
                do_sample=True,
                temperature=0.9,
                max_length=4096,
                max_new_tokens=50,
            )
            generated_token_ids = gen_tokens[0][input_ids.shape[1]:]
            generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
            generated_icl.append("New Fact: " + generated_text)

        return "\n\n".join(generated_icl)
    else:
        generated = [
            """New Fact: The location of the event that was attended by the creator of Garfield and Friends is the City of Rome.
Prompt: The event that was attended by the creator of Garfield and Friends was held in 
Thoughts: The location of the event that was attended by the creator of Garfield and Friends is the City of Rome. The City of Rome is the capital of Italy.
Answer: Rome""",
            """New Fact: The university where J. K. Rowling was educated is the University of Edinburgh. The headquarters of the University of Edinburgh is located in the city of Edinburgh.
Prompt: The headquarters of the institution where J. K. Rowling was educated is located in 
Thoughts: The university where J. K. Rowling was educated is the University of Edinburgh. The headquarters of the University of Edinburgh is located in the city of Edinburgh.
Answer: Edinburgh""",
            """New Fact: The city where the headquarters of The American Institute of Architecture are located is New York City. The American Institute of Architecture is a non-profit organization.
Prompt: The organization where the headquarters of the American Institute of Architecture are located is named 
Thoughts: The city where the headquarters of The American Institute of Architecture are located is New York City. The American Institute of Architecture is a non-profit organization.
Answer: American Institute of Architecture""",
            """New Fact: The work location of the creator of the television show Law & Order is New York City. The work location of the creator of Law & Order is New York City.
Prompt: The creator of the television show Law & Order in 
Thoughts: The work location of the creator of the television show Law & Order is New York City. The work location of the creator of Law & Order is New York City.
Answer: New York City""",
            """New Fact: The United States of America is the country whose capital is Washington, D.C.
Prompt: What is the name of the capital city of the country where the United States of America is located?
Thoughts: The United States of America is the country whose capital is Washington, D.C. The capital of the United States of America is Washington, D.C.
Answer: Washington, D.C."""
        ]
        return "\n\n".join(generated[:num_shot])


if __name__ == "__main__":
    pass
