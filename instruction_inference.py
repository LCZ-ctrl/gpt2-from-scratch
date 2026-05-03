import sys
import json
from pathlib import Path
import tiktoken
import torch

from config import GPT_SMALL, GPT_MEDIUM, GPT_LARGE, GPT_XL
from gpt import GPTModel
from generate import generate_text
from dataset import format_input, text_to_token_ids, token_ids_to_text
from instruction_finetune import LoRALayer, LinearWithLoRA, replace_linear_with_lora


def main(gpt_config, model_path, rank, alpha):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = tiktoken.get_encoding("gpt2")

    # Load fine-tuned weight into model
    model = GPTModel(gpt_config)
    replace_linear_with_lora(model, rank=rank, alpha=alpha)
    if not Path(model_path).exists():
        print(f"Error: model weight {model_path} does not exist, please fine-tune first!\n")
        sys.exit(1)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    print("Model loaded successfully.\n💬 Start conversation now (type 'exit' to quit)\n")
    # Interactive conversation loop
    while True:
        # Instruction input
        instruction = input("📌 Instruction: ").strip()
        if instruction.lower() == "exit":
            break
        if not instruction:
            print("Instruction cannot be empty!\n")
            continue

        # Optional extra input
        extra_input = input("📎 Task input (if necessary, press 'Enter' to skip): ").strip()
        entry = {"instruction": instruction, "input": extra_input if extra_input else None, "output": ""}
        input_text = format_input(entry)
        full_prompt = f"{input_text}\n<|assistant|>\n"

        # Convert the text prompt to token IDs
        input_ids = text_to_token_ids(full_prompt, tokenizer).to(device)
        # Generate response
        with torch.no_grad():
            output_ids = generate_text(
                model=model,
                device=device,
                idx=input_ids,
                max_new_tokens=512,
                context_size=1024,
                temperature=0.8,
                top_k=5,
                eos_id=50256
            )

        # Decode the entire token sequence back to text
        generated_text = token_ids_to_text(output_ids, tokenizer)
        response = generated_text[len(full_prompt):].strip()
        print(f"🤖 Model response: {response}\n")


if __name__ == "__main__":
    model_config = GPT_XL  # [GPT_SMALL, GPT_MEDIUM, GPT_LARGE, GPT_XL]
    model_size = model_config["size"]
    model_path = f"ckpt/finetune/gpt2-{model_size}/model-sft.pth"

    main(model_config, model_path, rank=16, alpha=16)
