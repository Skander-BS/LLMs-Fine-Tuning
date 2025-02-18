import os
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PEFT_MODEL_ID = f"skander-bs/Qwen2.5_1.5B_Reasoning"

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

def load_model_and_tokenizer():
    """Loads the base model and tokenizer, and applies the LoRA adapter."""
    
    config = PeftConfig.from_pretrained(PEFT_MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(PEFT_MODEL_ID)

    model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(model, PEFT_MODEL_ID)

    model.to(torch.bfloat16)
    model.eval()  

    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=300):
    """Generates a response given a prompt using the fine-tuned model."""
    
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}  # Move tensors to GPU
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.95,
        temperature=0.01,
        repetition_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)



def main():
    """Loads the model, prepares a prompt, and generates a response."""

    model, tokenizer = load_model_and_tokenizer()

    prompt = """<bos><start_of_turn>human
    You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.
    You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.
    
    Here are the available tools:<tools> 
    [{'type': 'function', 'function': {'name': 'convert_currency', 'description': 'Convert from one currency to another', 
    'parameters': {'type': 'object', 'properties': {'amount': {'type': 'number', 'description': 'The amount to convert'}, 
    'from_currency': {'type': 'string', 'description': 'The currency to convert from'}, 
    'to_currency': {'type': 'string', 'description': 'The currency to convert to'}}, 'required': ['amount', 'from_currency', 'to_currency']}}}, 
    
    {'type': 'function', 'function': {'name': 'calculate_distance', 'description': 'Calculate the distance between two locations', 
    'parameters': {'type': 'object', 'properties': {'start_location': {'type': 'string', 'description': 'The starting location'}, 
    'end_location': {'type': 'string', 'description': 'The ending location'}}, 'required': ['start_location', 'end_location']}}}] 
    </tools>

    Use the following pydantic model json schema for each tool call you will make: 
    {'title': 'FunctionCall', 'type': 'object', 'properties': {'arguments': {'title': 'Arguments', 'type': 'object'}, 
    'name': {'title': 'Name', 'type': 'string'}}, 'required': ['arguments', 'name']}
    
    For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
    <tool_call>
    {tool_call}
    </tool_call>
    
    Also, before making a call to a function take the time to plan the function to take. 
    Make that thinking process between <think>{your thoughts}</think>

    Hi, I need to convert 500 USD to Euros. Can you help me with that?<end_of_turn><eos>
    <start_of_turn>model
    <think>"""

    response = generate_response(model, tokenizer, prompt)

    print("\nGenerated Response:\n", response)


if __name__ == "__main__":
    main()