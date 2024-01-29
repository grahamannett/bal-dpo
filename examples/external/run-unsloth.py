import os
import torch

from transformers import TrainingArguments
from unsloth import FastLanguageModel

from trl import DPOTrainer
from examples.external.utils import apply_chat_template, get_datasets

# from unsloth import PatchDPOTrainer
#

model_name = "unsloth/zephyr-sft-bnb-4bit"
max_seq_length = 4096  # Choose any! We auto support RoPE Scaling internally!
torch_dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.
debug: bool = os.environ.get("DEBUG", "false").lower() == "true"
output_dir: str = "output"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,  # Choose ANY! eg mistralai/Mistral-7B-Instruct-v0.2
    max_seq_length=max_seq_length,
    dtype=torch_dtype,
    load_in_4bit=load_in_4bit,
)

raw_datasets = get_datasets(
    {"HuggingFaceH4/ultrafeedback_binarized": 0.005},  # 0.5% sampled
    splits=["train_prefs", "test_prefs"],
)

raw_datasets = raw_datasets.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer, "task": "dpo"},
    num_proc=12,
    remove_columns=list(raw_datasets["train"].features),
    desc="Formatting comparisons with prompt template",
)

# Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
for split in ["train", "test"]:
    raw_datasets[split] = raw_datasets[split].rename_columns(
        {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
    )

# trim dataset for debugging/quicker
train_dataset = raw_datasets["train"].select(range(10)) if debug else raw_datasets["train"]
train_dataset = raw_datasets["test"].select(range(10)) if debug else raw_datasets["test"]

model = FastLanguageModel.get_peft_model(
    model,
    r=32,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0,  # Currently only supports dropout = 0
    bias="none",  # Currently only supports bias = "none"
    use_gradient_checkpointing=True,
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)


training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_ratio=0.1,
    num_train_epochs=1,
    learning_rate=5e-6,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.0,
    lr_scheduler_type="linear",
    seed=42,
    output_dir=output_dir,
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    beta=0.1,
    train_dataset=train_dataset,
    # eval_dataset = raw_datasets["test"],
    tokenizer=tokenizer,
    max_length=1024,
    max_prompt_length=512,
)


dpo_trainer.train()
