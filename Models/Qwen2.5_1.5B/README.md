# Qwen 2.5-1.5B Fine Tuning

This part exposes how to fine tune Qwen model for function calling to optimize agents behaviour using LoRA (PEFT) Fine tuning technique.
This code example uses : Jofthomas/hermes-function-calling-thinking-V1 dataset optimized for thinking.

You can find my version in here : [skander-bs/Qwen2.5_1.5B_Reasoning](https://huggingface.co/skander-bs/Qwen2.5_1.5B_Reasoning)

## 1. Create a fresh new environment
Clone the repository :

```bash
git clone https://github.com/Skander-BS/LLMs-Fine-Tuning.git && cd LLMs-Fine-Tuning/Models/Qwen2.5_1.5B
```

It's recommended to use a virtual environment to avoid conflicts.

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

To start fine tuning, make sure to export your HF token as an env variable 

```bash
export HF_TOKEN=<xxx>
```

-Make sure to use your custom dataset or the dataset of your choice and make the necessary modifications.
-Customize the training hyperparameters as needed.


## 2 . Start Training

To start training, make sure you're in the current directory and everything is setup in the code and then execute :

```bash
python3 main.py
```

## 3 . Inference

To infer and test the model, make sure to update the HF Repository at the top and your prompt.
This part of the code merges the LoRA Adapeters to the base model for inference.

```bash
python3 infer.py
```

