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

def load_ripple(json_file):
    q_types = ["Logical_Generalization", "Compositionality_I", "Compositionality_II", "Subject_Aliasing", "Relation_Specificity"]
    print("Loading Ripple dataset...")
    with open(json_file, 'r') as f:
        dataset = json.load(f)
    
    output = []
    for i, entry in enumerate(tqdm.tqdm(dataset)):
        new_fact = entry["edit"]["prompt"]
        unformatted_questions = []
        for q_type in q_types:
            cur_type_questions = []
            for question in entry[q_type]:
                question["type"] = q_type
                cur_type_questions.append(question)
            unformatted_questions += cur_type_questions
        
        formatted_questions = []
        for question in unformatted_questions:
            questions = []
            for test_query in question["test_queries"]:
                questions.append("New Fact: " + new_fact + "\n" + "Prompt: " + test_query["prompt"] + "\n")
            if len(question["test_queries"][0]["answers"]) == 0:
                continue
            
            answers = question["test_queries"][0]["answers"][0]["aliases"]
            answers.append(question["test_queries"][0]["answers"][0]["value"])
            question["formated_questions"] = questions
            question["formated_answers"] = answers
            formatted_questions.append(question)
        output.extend(formatted_questions)
    return output
