import torch
import deepspeed

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments


torch_dtype = torch.bfloat16

model = AutoModelForCausalLM.from_pretrained(
    "adept/fuyu-8b",
    torch_dtype=torch_dtype,
)

model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args, model=model, model_parameters=params)
