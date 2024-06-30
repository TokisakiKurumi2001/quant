from transformers import AutoTokenizer
import json
from loguru import logger
from statistics import median, mean
from typing import Optional, List
from numpy import quantile

DATA_PATH="data_ift/train.jsonl"

def calc_stats(ls: List[int]):
    print(f'Median: {median(ls)}')
    print(f'Mean: {mean(ls)}')
    print(f'Max: {max(ls)}')
    print(f'Min: {min(ls)}')
    print(f'1st Quantile: {quantile(ls, 0.25)}')
    print(f'3rd Quantile: {quantile(ls, 0.75)}')
    print(f'90% Quantile: {quantile(ls, 0.9)}')

if __name__ == "__main__":
    data = []
    with open(DATA_PATH) as fin:
        for line in fin:
            _data = json.loads(line)
            data.append(_data)

    logger.info(f"Len of data: {len(data)}")
    # prompt
    prompts = [d['prompt'] for d in data]
    logger.debug(f'Prompts sample: {prompts[0]}')
    # full
    full = [d['prompt'] + d['output'] for d in data]
    logger.debug(f'Full sample: {full[0]}')
    
    logger.info(f'Loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained('llama_aqlm')

    prompts_len = [len(tokenizer.encode(p)) for p in prompts]

    logger.info(f'Statistics of prompt length')
    calc_stats(prompts_len)

    fulls_len = [len(tokenizer.encode(f)) for f in full]

    logger.info(f'Statistics of full length')
    calc_stats(fulls_len)