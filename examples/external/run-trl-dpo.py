# This is rough example of how TRL and DPO can be used together.
# NOTE: based on:
# - https://gist.github.com/lewtun/b9d46e00292d9ecdd6fd9628d53c2814
# - (unsloth example) https://github.com/unslothai/unsloth
# but it doesn't work exactly, what i changed was: remove device_map,

# TO RUN (TESTED):
# `accelerate launch --num_processes=2 --config_file conf/deepspeed-zero3.yaml examples/external/run-trl-dpo.py --use_peft`
#
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments
from trl import DPOTrainer

# imports from non external dependencies
from utils import apply_chat_template, get_datasets


debug: bool = os.environ.get("DEBUG", "false").lower() == "true"

accelerator = Accelerator()


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_name: Optional[str] = field(default="gpt2", metadata={"help": "the model name"})
    # have tried below but dont work, not clear if due to memory or model itself
    # model_name: Optional[str] = field(default="adept/fuyu-8b", metadata={"help": "the model name"})
    # model_name: Optional[str] = field(default="mistralai/Mistral-7B-Instruct-v0.2", metadata={"help": "the model name"})
    # model_name: Optional[str] = field(default="unsloth/zephyr-sft-bnb-4bit", metadata={"help": "the model name"})

    # remaining args are similar to the sft example
    dataset_name: Optional[str] = field(
        default="HuggingFaceH4/ultrafeedback_binarized", metadata={"help": "the dataset name"}
    )
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
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(
        default=1000, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


# DPO SPECIFIC - Need this tokenizer for DPO
# need this tokenizer for DPO
tokenizer = AutoTokenizer.from_pretrained("unsloth/zephyr-sft-bnb-4bit")

# Step 1: Load the dataset
# dataset = load_dataset(script_args.dataset_name, split="train[:500]")
raw_datasets = get_datasets(
    {"HuggingFaceH4/ultrafeedback_binarized": 0.005},  # 0.5% sampled
    splits=["train_prefs", "test_prefs"],
)

column_names = list(raw_datasets["train"].features)

raw_datasets = raw_datasets.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer, "task": "dpo"},
    num_proc=12,
    remove_columns=column_names,
    desc="Formatting comparisons with prompt template",
)

for split in ["train", "test"]:
    raw_datasets[split] = raw_datasets[split].rename_columns(
        {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
    )

train_dataset = raw_datasets["train"].select(range(10)) if debug else raw_datasets["train"]
test_dataset = raw_datasets["test"].select(range(10)) if debug else raw_datasets["test"]


# Step 2: Load the model
if script_args.load_in_8bit and script_args.load_in_4bit:
    raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
elif script_args.load_in_8bit or script_args.load_in_4bit:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
    )
    # Copy the model to each device
    # device_map = {"": int(os.environ.get("LOCAL_RANK", -1))} # {"": accelerator.local_process_index}
    torch_dtype = torch.bfloat16
else:
    device_map = None
    quantization_config = None
    torch_dtype = None

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    trust_remote_code=script_args.trust_remote_code,
    # torch_dtype=torch.bfloat16,  # torch_dtype,
    # device_map=device_map,
    # quantization_config=quantization_config,
    # load_in_4bit=True,  # doesnt work for DPO
)

# need ref model for DPO when not using unsloth
ref_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    # device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch.bfloat16,  # torch_dtype,
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
    optim="adamw_8bit",
    save_steps=script_args.save_steps,
    save_total_limit=script_args.save_total_limit,
    push_to_hub=False,
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


trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    beta=0.1,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    max_length=1024,
    max_prompt_length=512,
    peft_config=peft_config,
)


model, trainer = accelerator.prepare(model, trainer)
trainer.train()

# Step 6: Save the model
trainer.save_model(script_args.output_dir)
