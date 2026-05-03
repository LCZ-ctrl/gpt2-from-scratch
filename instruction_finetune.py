import torch
import tiktoken
import json
import time
import random
import math
from pathlib import Path
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.functional import cross_entropy
from torch.amp import autocast, GradScaler

from config import GPT_SMALL, GPT_MEDIUM, GPT_LARGE, GPT_XL
from dataset import InstructionDataset, format_input, custom_collate_fn
from gpt import GPTModel
from download import download_and_load_gpt2, load_weights_into_gpt


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()

        # Matrix A: (in_dim, rank)
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        # Matrix B: (rank, out_dim)
        # This preserves the original model's behavior at the start of training
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))

        self.alpha = alpha
        self.scaling = alpha / rank

    def forward(self, x):
        # h = (alpha / rank) * (x @ A @ B)
        x = self.scaling * (x @ self.A @ self.B)
        return x


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()

        # Keep the original linear layer unchanged
        self.linear = linear

        # Create a LoRA branch with the same input/output dimensions
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        # Output = original linear + LoRA adaptation
        return self.linear(x) + self.lora(x)


def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        # Replace each Linear layer with a LinearWithLoRA wrapper
        if isinstance(module, torch.nn.Linear):
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # Recursively process nested submodules
            replace_linear_with_lora(module, rank, alpha)


# Almost the same as "train_model_simple" in pretrain.py
def finetune_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs):
    train_losses, val_losses = [], []
    scaler = GradScaler("cuda") if device.type == "cuda" else None

    print("🚀 Start fine-tuning...")

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_train_loss = 0.0
        num_train_batches = 0

        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}/{num_epochs}")
        for input_batch, target_batch in pbar:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration

            if scaler:
                with autocast("cuda"):
                    logits = model(input_batch)
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

            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(loss=loss.item(), lr=f"{current_lr:.2e}")

        avg_train_loss = total_train_loss / num_train_batches

        # Evaluate model
        model.eval()
        total_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Val Epoch {epoch + 1}/{num_epochs}")
            for input_batch, target_batch in val_pbar:
                logits = model(input_batch)
                loss = cross_entropy(logits.flatten(0, 1), target_batch.flatten())
                total_val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = total_val_loss / num_val_batches

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        tqdm.write(f"Epoch {epoch + 1}: Train loss {avg_train_loss:.3f} | Val loss {avg_val_loss:.3f}")

    return train_losses, val_losses


def main(gpt_config, settings, data_path="data/alpaca_gpt4_data.json"):
    # ========== Prepare dataset ==========
    file_path = Path(data_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path.absolute()}")
    with open(file_path, "r") as f:
        data = json.load(f)
    print("Number of entries:", len(data))

    random.shuffle(data)

    train_portion = int(len(data) * 0.9)  # 90% for training, 10% for validation
    train_data = data[:train_portion]
    val_data = data[train_portion:]

    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))

    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu = torch.cuda.get_device_name(device)
        tqdm.write(f"💻 Device: {gpu}")
    else:
        tqdm.write("💻 Device: CPU")

    customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=512)

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings["batch_size"],
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=settings["num_workers"]
    )

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=settings["batch_size"],
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=settings["num_workers"]
    )

    # ========== Load pre-trained weight into model ==========
    model_size = gpt_config["size"]
    _, params = download_and_load_gpt2(model_size=model_size, models_dir="weights/gpt2")
    model = GPTModel(gpt_config)
    load_weights_into_gpt(model, params)
    model.to(device)
    model.eval()

    # ========== Implementing LoRA to reduce parameters ==========
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Total trainable parameters before: {total_params:,}")
    for param in model.parameters():
        param.requires_grad = False
    replace_linear_with_lora(model, rank=16, alpha=16)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total trainable parameters with LoRA: {total_params:,}")

    # ========== Fine-tuning the model ==========
    start_time = time.time()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )

    train_losses, val_losses = finetune_model_simple(
        model, train_loader, val_loader, optimizer, device, num_epochs=settings["num_epochs"]
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"\n✅ Training completed in {execution_time_minutes:.2f} minutes")

    # ========== Save the fine-tuned model ==========
    model_save_path = Path(f"ckpt/finetune/gpt2-{model_size}/model-sft.pth")
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"💾 Model saved to {model_save_path}")


if __name__ == "__main__":
    settings = {
        "learning_rate": 1e-4,
        "num_epochs": 2,
        "batch_size": 4,
        "num_workers": 0,
        "weight_decay": 0.01
    }
    model_config = GPT_XL  # [GPT_SMALL, GPT_MEDIUM, GPT_LARGE, GPT_XL]
    data_path = "data/alpaca_gpt4_data.json"

    main(model_config, settings, data_path)
