import json
import argparse
import os


argparser = argparse.ArgumentParser()
argparser.add_argument("json", type=str)

args = argparser.parse_args()

def is_answer_correct(provided_answers, correct_answers):
    return any(correct_answer.lower() in provided_answer.lower() for correct_answer in correct_answers for provided_answer in provided_answers)

total = {}
correct = {}

data = json.load(open(args.json))
for entry in data:
    answers = entry["formated_answers"]
    responses = entry["output_by_model"]
    q_type = entry["type"]
    
    correct_or_not = is_answer_correct(responses, answers)
    
    total[f"{q_type}"] = total.get(f"{q_type}", 0) + 1
    correct[f"{q_type}"] = correct.get(f"{q_type}", 0) + correct_or_not
    
    total[f"total"] = total.get(f"total", 0) + 1
    correct[f"total"] = correct.get(f"total", 0) + correct_or_not
    
print(os.path.basename(args.json))
for key in total.keys():
    if "total" not in key:
        print(f"{key}, accuracy: {correct[key]/total[key]} ({correct[key]}/{total[key]})")
print(f"Overall accuracy: {correct['total']/total['total']} ({correct['total']}/{total['total']})")