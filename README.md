# Pre-training-GPT-Model

This project provides end‑to‑end code for training a GPT‑style language model from scratch (or fine‑tuning an existing checkpoint) on large text corpora. It includes:

- Utilities for downloading and preprocessing data  
- A configurable training pipeline built on PyTorch  
- Scripts for evaluation and sample generation  

---

## Project Structure

```
pretraining‑gpt‑model/
├── README.md
├── LICENSE
├── requirements.txt
├── config/
│   ├── default.yaml          # Default hyperparameter settings
│   └── custom.yaml           # (Optional) user‑provided overrides
├── data/
│   ├── openwebtext/
|       ├── preprocess.py     # Download OpenWebText and tokenize it.
│       ├── train.bin         # Binary tokenized training data (generated file)
│       ├── val.bin           # Binary tokenized validation data (generated file)
│       └── meta.pkl          # Pickled metadata (vocab_size, etc.) (generated file)
│
├── model.py              # Model file 
├── train.py              # Main training entrypoint (supports DDP)
├── generate.py           # Sample/generate text from a trained checkpoint
│
├── logs/                     # TensorBoard logs and WandB runs (if enabled)
└── checkpoints/              # Saved checkpoints (ckpt.pt)
```

---

## Prerequisites

1. **Hardware**  
   - A GPU with ≥ 8 GB of VRAM is highly recommended for pretraining.  
   - If you plan to use mixed precision (float16 or bfloat16), ensure your GPU/driver support it.

2. **Software**  
   - **Python 3.8+** (<= 3.10)  
   - **CUDA 11.x** (for NVIDIA GPU users)  
   - **PyTorch 1.13+** with corresponding CUDA toolkit  
   - **git** (for cloning the repository)  

3. **Disk & Memory**  
   - At least **100 GB** of free disk space for dataset downloads and checkpoints  
   - ≥ 16 GB RAM (system memory) for data preprocessing and DDP setups  

---

## Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/thejesh-m/pretraining‑gpt‑model.git
   cd pretraining‑gpt‑model
   ```

2. **Create a Virtual Environment**  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Required Packages**  
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
---

## Dataset Preparation

Before training, you need binary tokenized data (`train.bin` and `val.bin`) plus a `meta.pkl` that stores the vocabulary size. This project expects data under `data/openwebtext/`.

1. **Download, Preprocess & Tokenize**  
   Convert raw text → token IDs → `.bin`. The simplest approach is to rely on a pretrained GPT‑2 tokenizer:

   ```bash
   python data/scriopenwebtextpts/preproces.py
   ```
   - `--tokenizer gpt2` uses Hugging Face’s GPT‑2 BPE tokenizer.  
   - The script will produce:  
     - `data/openwebtext/train.bin`  
     - `data/openwebtext/val.bin`  
     - `data/openwebtext/meta.pkl` (dictionary with `vocab_size`)

2. **Verify Data Files**  
   After preprocessing, you should see:
   ```
   data/openwebtext/
   ├── train.bin
   ├── val.bin
   └── meta.pkl
   ```
   - `train.bin` and `val.bin` are `uint16` NumPy memmap files containing token IDs.  
   - `meta.pkl` is a small pickle file that includes at least `{"vocab_size": <int>}`.

---

## Training the GPT Model

Once your data is ready, you can start training. The `train.py` script is fully configurable via `config/default.yaml` (or your own YAML override) and supports both single‑GPU and DDP.

### 1. Single‑GPU (Debug or Small‑Scale Run)

By default, `train.py` loads hyperparameters from `config/default.yaml`. To override specific fields, append `key=value` on the command line:

```bash
# Example: single‑GPU run with smaller batch size
python train.py batch_size=16 compile=False
```

- `batch_size=16` overrides the default micro‑batch size.  
- `compile=False` disables `torch.compile()`, which can be helpful for quick debugging.  

### 2. Multi‑GPU with DDP (Single Node)

To harness multiple GPUs on a single machine, use `torchrun` (PyTorch >= 1.9). For example, to train on 4 GPUs:

```bash
torchrun --standalone --nproc_per_node=4 train.py 
```

- `--nproc_per_node=4` spawns 4 processes, one per GPU.  
- `gradient_accumulation_steps=20` will be divided by `world_size` internally (so each GPU accumulates `20/4 = 5` micro-steps).  
- All other hyperparameters come from `config/default.yaml` unless overridden.

### 3. Multi‑Node DDP

If you have two nodes each with 4 GPUs (total 8 GPUs), you could do:

- **On Node 0 (rank 0)**:  
  ```bash
  torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 --master_addr="123.456.123.456" --master_port=1234 train.py
  ```
- **On Node 1 (rank 1)**:  
  ```bash
  torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 --master_addr="123.456.123.456" --master_port=1234 train.py
  ```
  - If your cluster lacks InfiniBand, prefix each command with `NCCL_IB_DISABLE=1`.



### 4. Checkpoints & Logging

- Checkpoints are saved under `checkpoints/ckpt.pt` (or `out/ckpt.pt` if you renamed `out_dir`).  
- If `wandb_log: true` in your config (or overridden), metrics (train/val loss, LR, MFU) will appear in your Weights & Biases dashboard.

---

## Evaluating the Model Performance

### Sampling Outputs

After training completes (or at any saved checkpoint), you can generate text samples (greedy or with top‑k/top‑p sampling) using `generate.py`:

```bash
python generate.py --model_path checkpoints/ckpt.pt --prompt "Once upon a time, in a land far away" --max_new_tokens 100 --temperature 0.8 --top_k 50 --top_p 0.95
```

- `--prompt`: initial text to condition on  
- `--max_new_tokens`: number of tokens to sample beyond the prompt  
- `--temperature`: sampling temperature (0 = greedy, 1 = raw logits)  
- `--top_k` / `--top_p`: control nucleus sampling  

The script prints the generated continuation to stdout.

---

That’s it! You now have a fully working pipeline for pretraining, evaluating, and sampling from a custom GPT‑style language model.
