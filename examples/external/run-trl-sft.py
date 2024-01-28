#   - Run from root of trl repo with: accelerate launch --config_file=examples/accelerate_configs/deepspeed_zero3.yaml --gradient_accumulation_steps 8 examples/scripts/sft_trainer.py
# NOTE: from
# https://gist.github.com/lewtun/b9d46e00292d9ecdd6fd9628d53c2814
# but it doesn't work exactly, what i changed was: remove device_map,

# TO RUN (TESTED):
# `accelerate launch --num_processes=2 --config_file conf/deepspeed-zero3.yaml external/run-trl-sft.py --use_peft`
#
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments

from trl import DPOTrainer, SFTTrainer

tqdm.pandas()


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    # model_name: Optional[str] = field(default="mistralai/Mistral-7B-v0.1", metadata={"help": "the model name"})
    # model_name: Optional[str] = field(default="adept/fuyu-8b", metadata={"help": "the model name"})
    model_name: Optional[str] = field(default="mistralai/Mistral-7B-Instruct-v0.2", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(default="stingning/ultrachat", metadata={"help": "the dataset name"})
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    # log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    log_with: Optional[str] = field(default="none", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=2.0e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=2, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=2, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=5, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(
        default=1000, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Step 1: Load the dataset
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
# dataset = load_dataset(script_args.dataset_name, split="train[:20000]")
dataset = load_dataset(script_args.dataset_name, split="train[:500]")
dataset = dataset.train_test_split(test_size=0.1)


def prepare_dialogue(example):
    text = ""
    for idx, msg in enumerate(example["data"]):
        if idx % 2 == 0:
            text += f"<|user|>\n{msg}{tokenizer.eos_token}\n"
        else:
            text += f"<|assistant|>\n{msg}{tokenizer.eos_token}\n"
    example["text"] = text
    return example


dataset = dataset.map(prepare_dialogue, num_proc=8, remove_columns=["id", "data"])
accelerator = Accelerator()


# Step 2: Load the model
if script_args.load_in_8bit and script_args.load_in_4bit:
    raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
elif script_args.load_in_8bit or script_args.load_in_4bit:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
    )
    # Copy the model to each device
    # device_map = {"": Accelerator().local_process_index}
    # device_map = {"": accelerator.local_process_index}
    # device_map = {"": int(os.environ.get("LOCAL_RANK", -1))}
    torch_dtype = torch.bfloat16
else:
    device_map = None
    quantization_config = None
    torch_dtype = None

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=quantization_config,
    # device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
    use_auth_token=script_args.use_auth_token,
)


# Step 3: Define the training arguments
training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=True,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    num_train_epochs=script_args.num_train_epochs,
    max_steps=script_args.max_steps,
    report_to=script_args.log_with,
    save_steps=script_args.save_steps,
    save_total_limit=script_args.save_total_limit,
    bf16=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    evaluation_strategy="epoch",
    logging_first_step=True,
)

# Step 4: Define the LoraConfig
if script_args.use_peft:
    peft_config = LoraConfig(
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
    )
else:
    peft_config = None


train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Step 5: Define the Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    max_seq_length=script_args.seq_length,
    train_dataset=train_dataset,  # dataset["train"],
    eval_dataset=test_dataset,  # dataset["test"],
    dataset_text_field=script_args.dataset_text_field,
    peft_config=peft_config,
    packing=True,
)


model, trainer = accelerator.prepare(model, trainer)

trainer.train()

# Step 6: Save the model
trainer.save_model(script_args.output_dir)
