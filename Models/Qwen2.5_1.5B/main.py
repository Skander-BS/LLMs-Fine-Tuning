import os
import torch
from enum import Enum
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, TaskType


SEED = 42
set_seed(SEED)


HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN is not set. Please set it in your environment.")

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_NAME = "Jofthomas/hermes-function-calling-thinking-V1"


# Define Special Tokens
class ChatmlSpecialTokens(str, Enum):
    """Enum for special tokens used in chat formatting."""
    tools = "<tools>"
    eotools = "</tools>"
    think = "<think>"
    eothink = "</think>"
    tool_call = "<tool_call>"
    eotool_call = "</tool_call>"
    tool_response = "<tool_reponse>"
    eotool_response = "</tool_reponse>"
    pad_token = "<pad>"
    eos_token = "<eos>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


def setup_tokenizer():
    """Load tokenizer and set chat template."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenizer.chat_template = (
        "{{ bos_token }}"
        "{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}"
        "{% for message in messages %}"
        "{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] | trim + '<end_of_turn><eos>\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
    )

    tokenizer.add_special_tokens({"pad_token": ChatmlSpecialTokens.pad_token.value})
    tokenizer.add_special_tokens({"additional_special_tokens": ChatmlSpecialTokens.list()})

    return tokenizer


def preprocess(sample, tokenizer):
    """Preprocess dataset samples by applying chat formatting."""
    messages = sample["messages"]
    first_message = messages[0]

    if first_message["role"] == "system":
        system_message_content = first_message["content"]
        messages[1]["content"] = (
            f"{system_message_content}Also, before making a call to a function, take the time to plan the function. "
            f"Make that thinking process between <think>{{your thoughts}}</think>\n\n"
            f"{messages[1]['content']}"
        )
        messages.pop(0)

    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


def load_and_process_dataset(tokenizer):
    """Load dataset and apply preprocessing."""
    dataset = load_dataset(DATASET_NAME)
    dataset = dataset.rename_column("conversations", "messages")
    dataset = dataset.map(lambda x: preprocess(x, tokenizer), remove_columns=["messages"])
    return dataset["train"].train_test_split(0.1)


def initialize_model(tokenizer):
    """Load pre-trained model and adjust embeddings."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        attn_implementation='eager',
        device_map="auto"
    )

    model.resize_token_embeddings(len(tokenizer))
    model.to(torch.bfloat16)
    return model


def setup_lora():
    """Configure LoRA fine-tuning parameters."""
    return LoraConfig(
        r=16,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=[
            "gate_proj", "q_proj", "lm_head", "o_proj", "k_proj", 
            "embed_tokens", "down_proj", "up_proj", "v_proj"
        ],
        task_type=TaskType.CAUSAL_LM
    )


def setup_training_args(output_dir):
    """Configure training arguments."""
    return SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        save_strategy="no",
        eval_strategy="epoch",
        logging_steps=5,
        learning_rate=1e-4,
        max_grad_norm=1.0,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        bf16=True,
        hub_private_repo=False,
        push_to_hub=False,
        num_train_epochs=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        packing=True,
        max_seq_length=1500,
    )


def main():
    """Main function to run fine-tuning."""
    
    username = "xxx"
    output_dir = "xxx"

    tokenizer = setup_tokenizer()

    dataset = load_and_process_dataset(tokenizer)

    model = initialize_model(tokenizer)

    peft_config = setup_lora()

    training_args = setup_training_args(output_dir)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    trainer.save_model()
    trainer.push_to_hub(f"{username}/{output_dir}")
    
    tokenizer.eos_token = "<eos>"
    tokenizer.push_to_hub(f"{username}/{output_dir}", token=True)


if __name__ == "__main__":
    main()