import torch
import tiktoken
import time
from tqdm import tqdm
from pathlib import Path
from torch.nn.functional import cross_entropy
from torch.amp import autocast, GradScaler

from config import GPT_SMALL, GPT_MEDIUM, GPT_LARGE, GPT_XL
from dataset import text_to_token_ids, token_ids_to_text, create_dataloader_v1
from gpt import GPTModel
from generate import generate_text


def generate_and_print_sample(model, tokenizer, device, start_context):
    """Generate text from a given starting context."""
    model.eval()

    # Determine the maximum context length supported by the model
    context_size = model.pos_emb.weight.shape[0]

    # Encode the starting text into token IDs
    # Shape: [1, n_tokens]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        # Generate new tokens
        # Shape: [1, n_tokens + 50]
        token_ids = generate_text(
            model=model,
            device=device,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
            temperature=0.0,
            top_k=None,
            eos_id=50256
        )

        # Decode the generated token IDs back into a text string
        # Shape: [n_tokens + 50,]
        decoded_text = token_ids_to_text(token_ids, tokenizer)

        # Print the generated text, replacing newline characters with spaces
        print(decoded_text.replace("\n", " "))

    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       start_context, tokenizer):
    train_losses, val_losses = [], []
    scaler = GradScaler("cuda") if device.type == "cuda" else None

    print("🚀 Start pre-training...")

    # Main training loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        num_train_batches = 0

        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}/{num_epochs}")
        for input_batch, target_batch in pbar:
            # Move tensors to the specified device (e.g., "cuda" or "cpu")
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration

            # Use AMP if device is cuda
            if scaler:
                with autocast("cuda"):
                    # Forward pass to get logits
                    logits = model(input_batch)
                    # Cross entropy expects input of shape [N, C] where C is the number of classes.
                    # We flatten the batch and sequence dimensions:
                    # logits.flatten(0, 1) -> [batch_size * seq_len, vocab_size]
                    # target_batch.flatten() -> [batch_size * seq_len,]
                    loss = cross_entropy(logits.flatten(0, 1), target_batch.flatten())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(input_batch)
                loss = cross_entropy(logits.flatten(0, 1), target_batch.flatten())
                loss.backward()
                optimizer.step()

            total_train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = total_train_loss / num_train_batches

        # Evaluate model
        model.eval()
        total_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Val Epoch {epoch + 1}/{num_epochs}")
            for input_batch, target_batch in val_pbar:
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)
                logits = model(input_batch)
                loss = cross_entropy(logits.flatten(0, 1), target_batch.flatten())
                total_val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = total_val_loss / num_val_batches

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        tqdm.write(f"Epoch {epoch + 1}: Train loss {avg_train_loss:.3f} | Val loss {avg_val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses


def main(gpt_config, settings, data_path="data/the-verdict.txt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_path = Path(data_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path.absolute()}")
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    model = GPTModel(gpt_config).to(device)
    model_size = gpt_config["size"]

    start_time = time.time()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )

    # Train/validation ratio
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=256,
        stride=128,
        drop_last=True,
        shuffle=True,
        num_workers=settings["num_workers"]
    )

    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=256,
        stride=128,
        drop_last=False,
        shuffle=False,
        num_workers=settings["num_workers"]
    )

    tokenizer = tiktoken.get_encoding("gpt2")

    train_losses, val_losses = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=settings["num_epochs"],
        start_context="Every effort makes you", tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"\n✅ Training completed in {execution_time_minutes:.2f} minutes")

    model_save_path = Path(f"ckpt/pretrain/gpt2-{model_size}/model.pth")
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"💾 Model saved to {model_save_path}")


if __name__ == "__main__":
    settings = {
        "learning_rate": 5e-4,
        "num_epochs": 10,
        "batch_size": 8,
        "num_workers": 0,
        "weight_decay": 0.1,
    }
    model_config = GPT_SMALL  # [GPT_SMALL, GPT_MEDIUM, GPT_LARGE, GPT_XL]
    data_path = "data/the-verdict.txt"

    main(model_config, settings, data_path)
