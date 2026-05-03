# Build a GPT-2 From Scratch

This repository contains my implementation of a GPT‑2 model following the concepts from the book <strong>*Build a Large
Language Model (From Scratch)*</strong>, including structure building, pretraining and
instruction fine-tuning.  
I'm using `PyTorch 2.10.0+cu128` in `Python 3.12.0`.

## Structure

```
├── data/                     # Pretrain & finetune data
|   └── ...
├── config.py                 # GPT config (Small / Medium / Large / XL)
├── dataset.py                # Tokenization, sliding window, format & collate
├── gpt.py                    # GPT model implementation
├── download.py               # Download and load official GPT-2 weights
├── generate.py               # Autoregressive text generation
├── pretrain.py               # Pre-training script
├── instruct_finetune.py      # Instruction fine-tuning script
└── instruct_inference.py     # Chat with fine‑tuned model
```

## Requirements

```
numpy==2.4.4
tensorflow==2.21.0
tiktoken==0.12.0
torch==2.10.0+cu128
tqdm==4.67.3
```

## Model Structure

An overview of the GPT model architecture:

<br>
<p align="center">
    <img src="./images/gpt model.png" width="500" />
    <br>
    <em><strong>GPT model</strong></em>
</p>
<br>

| Model      | Context Length | Embedding Dimension | Heads | Layers | Size  |
|:-----------|----------------|:--------------------|:------|--------|-------|
| GPT_SMALL  | 1024           | 768                 | 12    | 12     | 124M  |
| GPT_MEDIUM | 1024           | 1024                | 16    | 24     | 355M  |
| GPT_LARGE  | 1024           | 1280                | 20    | 36     | 774M  |
| GPT_XL     | 1024           | 1600                | 25    | 48     | 1558M |

## Pretrain

To pretrain the model, run the command -

```
python pretrain.py
```

Since pretraining is resource-heavy and time‑consuming, here for learning purposes, I just used the same tiny dataset
*the-verdict* from the book to pretrain *GPT_SMALL* (124M) and see how it performs. You can also use other datasets if you want.

## Fine-tuning

To fine-tune the pretrained model, run the command -

```
python instruction_finetune.py
```

If you haven't downloaded the official pretrained weight, it will download automatically the first time.  
Unlike the original book, I used *alpaca-gpt4* as my instruction fine‑tuning dataset, and I organized the data using
the *Phi‑3* prompt style and masked the instruction part when computing the loss.  

I used an NVIDIA GeForce RTX 4090 GPU (24GB VRAM) to fine‑tune *GPT_XL* (1.5B) for 2 epochs (about 169 minutes).
If you're using a smaller GPU, you can try increasing ```segments``` in ```gpt.py``` (gradient checkpointing), reducing
```batch_size``` or ```allowed_max_length``` in ```instruction_finetune.py```, or switching to a smaller model.

Once the fine-tuning process is complete, to play with your model, run the command -

```
python instruction_inference.py
```

The results are as follows -

```
Instruction: What is the opposite of "slow"?
Task input (if necessary, press 'Enter' to skip): 
Model response: The opposite of "slow" is "fast".

Instruction: Rewrite the sentence using a simile.
Task input (if necessary, press 'Enter' to skip): Her eyes are very beautiful.
Model response: Her eyes are like diamonds glittering in the sun.

Instruction: What is the plural of "mouse"?
Task input (if necessary, press 'Enter' to skip): 
Model response: The plural of "mouse" is "mice".

Instruction: Correct the spelling error in the sentence.
Task input (if necessary, press 'Enter' to skip): I will atend the meeting tomorrow.
Model response: I will attend the meeting tomorrow.
```

<br><br>
<em><strong>My fine-tuned weight:</strong></em> [GPT_XL](https://huggingface.co/LCZ-ctrl/LLMs-finetuned)
