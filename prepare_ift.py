import json, os, random

SAVE_DATA_PATH="data_ift"
DATA_PATH="data/dolly"

def prompt_template(item):
    if "input" not in item or len(item["input"]) == 0:
        template = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n"
        )
        prompt = template.format(instruction=item["instruction"])
    else:
        template = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
        )
        prompt = template.format(instruction=item["instruction"], input=item["input"])
            
    response = item["output"]
    return {'prompt': prompt, 'output': response}

if __name__ == "__main__":
    data = []
    with open(DATA_PATH+"/raw.jsonl") as fin:
        for line in fin:
            d = json.loads(line)
            data.append(d)

    # random shuffle
    random.shuffle(data)
    # first 5k for test, rest for training
    ds = [prompt_template(d) for d in data]

    test_ds = ds[:5000]
    train_ds = ds[5000:]

    os.makedirs(SAVE_DATA_PATH, exist_ok=True)
    with open(SAVE_DATA_PATH+"/train.jsonl", 'w+') as fout:
        for d in train_ds:
            fout.write(json.dumps(d) + "\n")

    with open(SAVE_DATA_PATH+"/test.jsonl", 'w+') as fout:
        for d in test_ds:
            fout.write(json.dumps(d) + "\n")