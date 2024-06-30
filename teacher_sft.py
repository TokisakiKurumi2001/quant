from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType
from peft import get_peft_model
from datasets import Dataset

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from time import sleep

from loguru import logger
import json, copy
from collections.abc import Mapping

DATA_PATH="./" # data/cnn/random/"
MODEL_PATH="llama_teacher"
TOKENIZER_PATH="llama_teacher"
MAX_LENGTH=1024
MAX_PROMPT_LENGTH=860
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEP=8
NUM_EPOCHS=5
SAVE_DIR="teacher-sft-ift"

def get_data(split: str):
    """
    Read JSONL -> convert HF datasets
    """
    data = []
    filename = DATA_PATH + split + ".jsonl"
    with open(filename, encoding="utf-8") as fin:
        for line in fin:
            _data = json.loads(line)
            data.append(_data)
    dataset = Dataset.from_list(data)
    return dataset

def tokenize_data(dataset: Dataset, tokenizer: AutoTokenizer):
    def tokenize_example(example):
        # code from `https://github.com/ZhengxiangShi/InstructionModelling/blob/main/src/finetune_kl.py#L262`
        example_text = [p + o for p, o in zip(example['prompt'], example['output'])]
        tokenized_example = tokenizer(example_text, max_length=MAX_LENGTH, truncation=True, padding="max_length", return_tensors='pt')
        input_ids = tokenized_example.input_ids
        labels = input_ids.clone() # copy.deepcopy(input_ids)
        labels[:, :-1] = labels[:, 1:].clone()
        labels[:, -1] = tokenizer.eos_token_id
        tokenized_prompt = tokenizer(example['prompt'], max_length=MAX_PROMPT_LENGTH, truncation=True, padding="max_length", return_tensors='pt')
        # mask the prompt part for avoiding loss
        labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
        attention_mask = tokenized_example.attention_mask 
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }
    tokenized_data = dataset.map(tokenize_example, batched=True, remove_columns=dataset.column_names,)
    return tokenized_data

def collate_fn(examples):
    if isinstance(examples, (list, tuple)) and isinstance(examples[0], Mapping):
        encoded_inputs = {key: [example[key] for example in examples] for key in examples[0].keys()}
    else:
        encoded_inputs = examples

    batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in encoded_inputs.items()}
    return batch

if __name__ == "__main__":
    logger.info("Loading data ...")
    train_ds = get_data('testing') # 'train')
    logger.success(f"There are {len(train_ds)} examples.")
    
    logger.info('Loading model ...')
    quantized_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")#, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    logger.success(f'Successfully load {MODEL_PATH} model')

    logger.info("Tokenizing data ...")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_tokenized_data = tokenize_data(train_ds, tokenizer)
    # breakpoint()

    # create dataloader
    train_dataloader = DataLoader(train_tokenized_data, batch_size=TRAIN_BATCH_SIZE, num_workers=4, shuffle=True, collate_fn=collate_fn)

    logger.info('Adding LoRA ...')
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.0, target_modules=['q_proj', 'v_proj'])
    quantized_model = get_peft_model(quantized_model, peft_config)
    quantized_model.print_trainable_parameters()

    # prepare optimizer and loss function
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(quantized_model.parameters(), lr=1e-4, )#fused=True)

    device = quantized_model.device
    vocabulary_size = quantized_model.config.vocab_size

    logger.info('Training ...')
    for epoch in range(NUM_EPOCHS):
        with tqdm(enumerate(train_dataloader), unit="batch", total=len(train_dataloader)) as tepoch:
            for step, batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop('labels')

                logits = quantized_model(**batch).logits
                loss = loss_fn(logits.view(-1, vocabulary_size), labels.view(-1))
                loss = loss / GRADIENT_ACCUMULATION_STEP

                loss.backward()
                
                if (step + 1) % GRADIENT_ACCUMULATION_STEP == 0:
                    optimizer.step()
                    optimizer.zero_grad()
         
                tepoch.set_postfix(loss=loss.item())
                sleep(0.1)

    logger.info(f"Saving to {SAVE_DIR}/")
    quantized_model.save_pretrained(SAVE_DIR)
