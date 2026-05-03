import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader


def text_to_token_ids(text, tokenizer):
    """Convert a text string into a tensor of token IDs with a batch dimension."""
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor  # shape: [1, n_tokens]


def token_ids_to_text(token_ids, tokenizer):
    """Convert a tensor of token IDs back into a text string."""
    flat = token_ids.squeeze(0)  # remove batch dimension
    text = tokenizer.decode(flat.tolist())
    return text  # shape: [n_tokens,]


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the text into overlapping sequences of max_length
        # Each sample consists of an input chunk and a target chunk (shifted by 1 to the right)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers
    )

    return dataloader


def format_input(entry):
    instruction_text = (
        f"<|user|>\n{entry['instruction']}"
    )

    input_text = f"\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # Separate list for instruction lengths
        self.instruction_lengths = []
        # Pre-tokenize texts
        self.encoded_texts = []

        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n<|assistant|>\n{entry['output']}"
            full_text = instruction_plus_input + response_text

            # Encode input and response
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

            # Collect instruction lengths
            instruction_length = len(tokenizer.encode(instruction_plus_input))
            self.instruction_lengths.append(instruction_length)

    def __getitem__(self, index):
        # Return both instruction lengths and texts separately
        return self.instruction_lengths[index], self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def custom_collate_fn(
        batch,
        pad_token_id=50256,
        ignore_index=-100,
        allowed_max_length=None,
        device="cpu"
):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item) + 1 for instruction_length, item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for instruction_length, item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        # Truncate the last token for inputs
        inputs = torch.tensor(padded[:-1])
        # Shift +1 to the right for targets
        targets = torch.tensor(padded[1:])

        # Replace all but the first padding tokens in targets by ignore_index
        mask = (targets == pad_token_id)
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # Mask all input and instruction tokens in the targets
        targets[:instruction_length - 1] = ignore_index

        # Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor
