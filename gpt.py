import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.eps = 1e-5  # to avoid division by zero
        self.scale = nn.Parameter(torch.ones(emb_dim))  # learnable parameter initialized with ones
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # learnable parameter initialized with zeros

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0  # d_out must be divisible by num_heads

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # dimension of each head

        # Project input to Q, K, V
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Output projection to merge head information
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # Buffer for the causal mask (upper triangular matrix)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, seq_len, d_in = x.shape

        # ----- Step 1: Project inputs to Q, K, V -----
        ## Shape: [b, seq_len, d_out]
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # ----- Step 2: Reshape into multiple heads -----
        ## We split the last dimension [d_out] into [num_heads, head_dim]
        ## Shape: [b, seq_len, num_heads, head_dim]
        keys = keys.view(b, seq_len, self.num_heads, self.head_dim)
        values = values.view(b, seq_len, self.num_heads, self.head_dim)
        queries = queries.view(b, seq_len, self.num_heads, self.head_dim)

        # ----- Step 3: Transpose matrix -----
        ## This allows batched matrix multiplication across heads.
        ## Shape: [b, num_heads, seq_len, head_dim]
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # ----- Step 4: Compute scaled dot-product attention scores -----
        ## Attention scores: softmax(QK^T / sqrt(head_dim)) * V
        ## Shape: [b, num_heads, seq_len, seq_len]
        attn_scores = queries @ keys.transpose(2, 3)

        ## Original mask truncated to seq_len and converted to boolean
        mask_bool = self.mask.bool()[:seq_len, :seq_len]

        ## Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        ## Normalize scores to weights using Softmax
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        ## Apply attention weights to values
        ## Shape: [b, seq_len, num_heads, head_dim]
        context_vec = (attn_weights @ values).transpose(1, 2)

        # ----- Step 8: Combine heads by transposing and reshaping -----
        ## Combine heads, where self.d_out = self.num_heads * self.head_dim
        ## Shape: [b, seq_len, d_out]
        context_vec = context_vec.contiguous().view(b, seq_len, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.layers = nn.Sequential(
            # Expand the embedding dimension by a factor of 4
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            # GELU activate function to introduce non-linearity
            GELU(),
            # Project the expanded representation back
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Multi-head self attention
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )

        # Feed forward network
        self.ff = FeedForward(cfg)

        # Layer normalizations (pre-norm)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

        # Dropout layer before added to shortcut
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)  # pre-norm
        x = self.att(x)  # shape: [batch_size, seq_len, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)  # shape: [batch_size, seq_len, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # add the original input back

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Token embedding layer: maps token IDs to vectors
        # Shape: [batch_size, seq_len] -> [batch_size, seq_len, emb_dim]
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])

        # Absolute positional embedding layer (learned)
        # Shape: [seq_len,] -> [seq_len, emb_dim]
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        # Shape: [batch_size, seq_len, emb_dim]
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Stack of transformer blocks (each contains multi-head attention + feed-forward)
        # Shape: [batch_size, seq_len, emb_dim]
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # Final layer normalization before the output head
        # Shape: [batch_size, seq_len, emb_dim]
        self.final_norm = LayerNorm(cfg["emb_dim"])

        # Output linear layer: projects from embedding dimension to vocabulary size
        # Shape: [batch_size, seq_len, emb_dim] -> [batch_size, seq_len, vocab_size]
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape  # [batch_size, seq_len]

        # Token embedding: [batch_size, seq_len, emb_dim]
        tok_embeds = self.tok_emb(in_idx)

        # Positional embedding: [seq_len, emb_dim]
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        # Combine embeddings (Element-wise addition)
        # Shape: [batch_size, seq_len, emb_size]
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)

        # Pass through all transformer blocks
        if self.training:
            # Use gradient checkpointing during training to save memory
            x = checkpoint.checkpoint_sequential(self.trf_blocks, segments=4, input=x, use_reentrant=False)
        else:
            x = self.trf_blocks(x)

        # Final layer normalization
        x = self.final_norm(x)

        # Project to vocabulary logits
        # Shape: [batch_size, seq_len, vocab_size]
        logits = self.out_head(x)
        return logits
