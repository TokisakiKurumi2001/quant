from datasets import load_dataset
import argparse, os, json
from loguru import logger
from typing import List, Dict

def convert_conversation_to_prompt(ls: List[Dict]) -> Dict:
    prompt = ls[0]['content']
    output = ls[1]['content']
    return {'prompt': prompt, 'output': output}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=str)
    args = parser.parse_args()

    dataset = load_dataset('tomg-group-umd/GenQA', 'general')

    path = 'genqa_full'
    if args.size == 'small':
        dataset = dataset.shuffle().select(range(55_000))
        path = 'genqa_small'

    os.makedirs(path, exist_ok=True)

    dataset = dataset.train_test_split(test_size=0.1)
    for split in ['train', 'test']:
        data = []
        data_split = dataset[split]
        for d in data_split:
            data.append(convert_conversation_to_prompt(d['text']))
        
        with open(path + '/' + split + '.jsonl', 'w+') as fout:
            for d in data:
                fout.write(json.dumps(d) + "\n")

        logger.info(f'Accumulate {len(data)} samples in {split} set.')