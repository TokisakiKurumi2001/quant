from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType
from peft import get_peft_model

from loguru import logger

DATA_PATH="data"
MODEL_PATH="llama_aqlm"
TOKENIZER_PATH="llama_aqlm"

if __name__ == "__main__":
    logger.info('Loading model ...')
    quantized_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype="auto", device_map="auto", low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    logger.success(f'Successfully load {MODEL_PATH} model')

    logger.info('Inference ...')
    output = quantized_model.generate(tokenizer("chicken", return_tensors="pt")["input_ids"].cuda(), min_new_tokens=128, max_new_tokens=128)
    logger.debug(tokenizer.decode(output[0]))

    logger.info('Adding LoRA ...')
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["layers.28.self_attn.v_proj"])
    quantized_model = get_peft_model(quantized_model, peft_config)
    quantized_model.print_trainable_parameters()

    logger.info('Inference ...')
    output = quantized_model.generate(tokenizer("chicken", return_tensors="pt")["input_ids"].cuda(), min_new_tokens=128, max_new_tokens=128)

    logger.debug(tokenizer.decode(output[0]))