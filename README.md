## LLM 

### Description  
This is a simple transformer-based text generation model using PyTorch. It processes a vocabulary file (`vocab.txt`), encodes text into integers, and trains a model using a small transformer architecture.  

### Features  
- Uses PyTorch (`torch.nn`) for model training  
- Implements a basic transformer with embedding and attention layers  
- Loads and processes a vocabulary file  
- Trains on a dataset with configurable hyperparameters  

### Requirements  
- Python 3.8+  
- PyTorch  
- `vocab.txt` file for training  

## Usage  
```bash
python gpt-v1.py --batch_size 32 --max_iters 200
