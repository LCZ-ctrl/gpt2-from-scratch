import torch
import tiktoken

from config import GPT_SMALL, GPT_MEDIUM, GPT_LARGE, GPT_XL
from gpt import GPTModel
from download import download_and_load_gpt2, load_weights_into_gpt
from dataset import text_to_token_ids, token_ids_to_text


def generate_text(model, device, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """Autoregressively text generation with temperature scaling and top-k sampling."""
    idx = idx.to(device)

    # Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:].to(device)
        with torch.no_grad():
            logits = model(idx_cond)  # shape: [batch_size, seq_len, vocab_size]
        logits = logits[:, -1, :]  # shape: [batch_size, vocab_size]

        # Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, top_indices = torch.topk(logits, top_k)  # [batch_size, top_k]
            min_val = top_logits[:, -1]
            # Mask logits smaller than the k-th largest value
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(device),
                logits
            )

        # Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # [batch_size, vocab_size]
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]
        # Otherwise, get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # [batch_size, 1]

        # Stop generating early if end-of-sequence token is encountered and eos_id is specified
        if eos_id is not None and idx_next.item() == eos_id:
            break

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # [batch_size, current_len+1]

    return idx


def main(gpt_config, input_prompt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained weight into model
    model_size = gpt_config["size"]
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="weights/gpt2")
    model = GPTModel(gpt_config)
    load_weights_into_gpt(model, params)
    model.to(device)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = generate_text(
        model=model,
        device=device,
        idx=text_to_token_ids(input_prompt, tokenizer),
        max_new_tokens=30,
        context_size=gpt_config["context_length"],
        top_k=5,
        temperature=0.8,
        eos_id=50256
    )
    output_text = token_ids_to_text(token_ids, tokenizer)
    print("\n📝 Output text:\n", output_text)


if __name__ == "__main__":
    model_config = GPT_XL  # [GPT_SMALL, GPT_MEDIUM, GPT_LARGE, GPT_XL]
    input_prompt = "Every effort makes you"

    main(model_config, input_prompt)
