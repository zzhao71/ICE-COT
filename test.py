import json
import random
from dotenv import load_dotenv
load_dotenv(".env")
import time
from openai import OpenAI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").to(device)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

prompt = ""
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
print(generated_text)