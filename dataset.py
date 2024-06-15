import tqdm
import json

def load_mquake(json_file):
    
    print("Loading MQuAKE dataset...")
    with open(json_file, 'r') as f:
        dataset = json.load(f)
    output = []
    for i, entry in enumerate(tqdm.tqdm(dataset)):
        new_facts = []
        for rewrite in entry["requested_rewrite"]:
            new_facts.append(rewrite["prompt"].format(rewrite["subject"]) + " " + rewrite["target_new"]["str"])
        questions = []
        for question in entry['questions']:
        
            question =  "New Fact: " + ", ".join(new_facts) + "\n" + "Question: " + question + "\n"
            questions.append(question)
        
        answers = entry["new_answer_alias"]
        answers.append(entry["new_answer"])

        entry["formated_questions"] = questions
        entry["formated_answers"] = answers
        output.append(entry)
    return output

        
