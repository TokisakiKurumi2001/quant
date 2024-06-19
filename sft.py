from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType
from peft import get_peft_model
from datasets import Dataset

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from loguru import logger
import json

DATA_PATH="data/cnn/random/"
MODEL_PATH="llama_aqlm"
TOKENIZER_PATH="llama_aqlm"
MAX_LENGTH=1024
MAX_PROMPT_LENGTH=512
TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEP=1
NUM_EPOCHS=2
SAVE_DIR="test"

def get_data(split: str):
    """
    Read JSONL -> convert HF datasets
    """
    data = []
    filename = DATA_PATH + split + ".json"
    with open(filename, encoding="utf-8") as fin:
        for line in fin:
            _data = json.loads(line)
            data.append(_data)
    dataset = Dataset.from_list(data)
    return dataset

def tokenize_data(dataset: Dataset, tokenizer: AutoTokenizer):
    def tokenize_example(example):
        # code from `https://github.com/ZhengxiangShi/InstructionModelling/blob/main/src/finetune_kl.py#L262`
        example_text = example['prompt'] + example['output'] + tokenizer.eos_token
        tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=MAX_LENGTH, truncation=True)
        input_ids = tokenized_example.input_ids
        labels = input_ids.clone()
        tokenized_prompt = tokenizer(example['prompt'], return_tensors='pt', max_length=MAX_PROMPT_LENGTH, truncation=True)
        # mask the prompt part for avoiding loss
        labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
        attention_mask = torch.ones_like(input_ids)
        return {
            'input_ids': input_ids.flatten(),
            'labels': labels.flatten(),
            'attention_mask': attention_mask.flatten(),
        }
    tokenized_data = dataset.map(tokenize_example, batched=True)
    return tokenize_data

if __name__ == "__main__":
    logger.info("Loading data ...")
    train_ds = get_data('train')
    logger.success(f"There are {len(train_ds)} examples.")
    
    logger.info('Loading model ...')
    quantized_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype="auto", device_map="auto", low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    logger.success(f'Successfully load {MODEL_PATH} model')

    logger.info("Tokenizing data ...")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding = "left"
    train_tokenized_data = tokenize_data(train_ds, tokenizer)

    # create dataloader
    train_dataloader = DataLoader(train_tokenized_data, batch_size=TRAIN_BATCH_SIZE, num_workers=32, shuffle=True)

    logger.info('Adding LoRA ...')
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["layers.28.self_attn.v_proj"])
    quantized_model = get_peft_model(quantized_model, peft_config)
    quantized_model.print_trainable_parameters()

    # prepare optimizer and loss function
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(quantized_model.parameters(), lr=1e-4, fused=True)

    device = quantized_model.device
    vocabulary_size = quantized_model.config.vocab_size

    logger.info('Training ...')
    for epoch in NUM_EPOCHS:
        with tqdm(enumerate(train_dataloader), unit="batch") as tepoch:
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