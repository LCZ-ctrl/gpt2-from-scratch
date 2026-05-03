GPT_SMALL = {
    "vocab_size": 50257,  # vocabulary size
    "context_length": 1024,  # context length
    "emb_dim": 768,  # embedding dimension
    "n_heads": 12,  # number of attention heads
    "n_layers": 12,  # number of layers
    "drop_rate": 0.1,  # dropout rate
    "qkv_bias": True,  # query-key-value bias
    "size": "124M"  # model size
}

GPT_MEDIUM = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 24,
    "drop_rate": 0.1,
    "qkv_bias": True,
    "size": "355M"
}

GPT_LARGE = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1280,
    "n_heads": 20,
    "n_layers": 36,
    "drop_rate": 0.1,
    "qkv_bias": True,
    "size": "774M"
}

GPT_XL = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1600,
    "n_heads": 25,
    "n_layers": 48,
    "drop_rate": 0.1,
    "qkv_bias": True,
    "size": "1558M"
}
