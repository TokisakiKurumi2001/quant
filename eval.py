from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType
from peft import get_peft_model

from loguru import logger
import json
from tqdm import tqdm

DATA_PATH="data/cnn/"
MODEL_PATH="llama_aqlm"
TOKENIZER_PATH="llama_aqlm"
LORA_DIR="test"
MAX_PROMPT_LENGTH=860
MAX_LENGTH=1024
NEW_TOKENS=MAX_LENGTH - MAX_PROMPT_LENGTH
OUTPUT_FILE='pred.jsonl'

def load_eval_data():
    # read data
    logger.info("Loading evaluation data ...")
    data = []
    filename = DATA_PATH + "test.jsonl"
    with open(filename) as fin:
        for line in fin:
            # line = line[:-1] # exclude the last \n line
            _data = json.loads(line)
            data.append(_data)

    logger.success(f"Sample {len(data)} samples.")
    return data

def evaluate(data, tokenizer, model, num_eval: int):
    # into the loop of tokenizer, generate, accumulate the result
    preds = []
    device = model.device
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    for _data in tqdm(data[:num_eval], desc="Evaluating"):
        text = _data['prompt']
        inputs = tokenizer(text, max_length=MAX_PROMPT_LENGTH, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model.generate(**inputs, max_new_tokens=NEW_TOKENS)
        out = tokenizer.decode(outputs[0], skip_special_tokens=True)
        preds.append(out)
    return preds

if __name__ == "__main__":
    eval_data = load_eval_data()

    logger.info('Loading model ...')
    quantized_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype="auto", device_map="auto", low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    logger.success(f'Successfully load {MODEL_PATH} model')

    # logger.info('Inference before LoRA ...')
    # output = quantized_model.generate(tokenizer("chicken", return_tensors="pt")["input_ids"].cuda(), min_new_tokens=128, max_new_tokens=128)
    # logger.debug(tokenizer.decode(output[0]))

    logger.info('Loading LoRA ...')
    quantized_model.load_adapter(LORA_DIR)

    # logger.info('Inference after LoRA ...')
    # output = quantized_model.generate(tokenizer("chicken", return_tensors="pt")["input_ids"].cuda(), min_new_tokens=128, max_new_tokens=128)
    # logger.debug(tokenizer.decode(output[0]))

    preds = evaluate(eval_data, tokenizer, quantized_model)
    with open(OUTPUT_FILE, 'w+') as fout:
        for p in preds:
            json_obj = {'predict': p}
            fout.write(json.dumps(json_obj) + "\n")
