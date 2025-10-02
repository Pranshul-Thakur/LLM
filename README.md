# GPT-Transformer

A lightweight GPT-style transformer language model built from scratch using PyTorch with character-level tokenization and Flash Attention optimization.

## Features

- Multi-head self-attention mechanism
- Configurable transformer architecture
- Flash Attention for efficient computation
- Character-level text generation
- Memory-mapped file reading for large datasets

## Requirements

```bash
pip install torch transformers flash-attn
```

## Usage

### Training

Prepare your data files:
- `vocab.txt` - Training corpus
- `train_split.txt` - Training data
- `val_split.txt` - Validation data

Run the notebook or convert to script:
```bash
python gpt-v1.py
```

### Hyperparameters

```python
batch_size = 32
block_size = 128
max_iters = 200
learning_rate = 2e-5
n_embd = 384
n_head = 4
n_layer = 4
dropout = 0.2
```

### Generate Text

```python
prompt = 'Your prompt here'
context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
generated = model.generate(context.unsqueeze(0), max_new_tokens=100)
print(decode(generated[0].tolist()))
```

## Architecture

- Token and positional embeddings
- 4 transformer blocks with multi-head attention
- Feed-forward networks with residual connections
- Layer normalization

## License

MIT License
