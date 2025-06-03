"""
Sample from a trained GPT model (loaded from a local checkpoint).
Usage example:
  python generate.py \
    --out_dir out \
    --start "Once upon a time," \
    --num_samples 5 \
    --max_new_tokens 200 \
    --temperature 0.8 \
    --top_k 50 \
    --device cuda
"""

import os
import argparse
import pickle
from contextlib import nullcontext

import torch
import tiktoken

from model import GPTConfig, GPT


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text samples from a trained GPT checkpoint")
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory containing 'ckpt.pt'.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="\n",
        help=(
            "Initial prompt string. "
            "Prefix with 'FILE:' to read prompt from a file, e.g. 'FILE:prompt.txt'."
        ),
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of independent samples to generate.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=500,
        help="Maximum number of tokens to generate per sample.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature: 1.0 = no change, <1.0 = less random, >1.0 = more random",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=200,
        help="Keep only top_k tokens with highest probability at each step.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on: 'cpu', 'cuda', 'cuda:0', etc.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
        help="Data type for autocast. Defaults to 'bfloat16' if supported, else 'float16'.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="If set, compile the model with torch.compile() (requires PyTorch 2.0+).",
    )
    return parser.parse_args()


def load_model_from_checkpoint(out_dir, device, compile_flag):
    """
    Load GPT from a local checkpoint in out_dir/ckpt.pt.
    Returns (model, checkpoint_data).
    """
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint["model_args"]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    # Remove any unwanted prefixes (e.g. _orig_mod.)
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)

    if compile_flag:
        model = torch.compile(model)

    return model, checkpoint


def setup_tokenizer(checkpoint):
    """
    Return (encode_fn, decode_fn). If a meta.pkl with 'stoi'/'itos' exists
    in the dataset directory recorded in checkpoint['config']['dataset'],
    use that; otherwise default to GPT-2 BPE.
    """
    cfg = checkpoint.get("config", {})
    dataset_name = cfg.get("dataset", None)
    if dataset_name:
        meta_path = os.path.join("data", dataset_name, "meta.pkl")
        if os.path.isfile(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            stoi, itos = meta["stoi"], meta["itos"]
            encode = lambda s: [stoi.get(ch, stoi.get("<|endoftext|>", 0)) for ch in s]
            decode = lambda lst: "".join([itos[i] for i in lst])
            return encode, decode

    # Fallback: use GPT-2 BPE via tiktoken
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda lst: enc.decode(lst)
    return encode, decode


def main():
    args = parse_args()

    # Determine device
    device = args.device
    if "cuda" in device and not torch.cuda.is_available():
        print("WARNING: CUDA not available, switching to CPU")
        device = "cpu"

    # Determine dtype / autocast context
    if args.dtype == "bfloat16":
        if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
            print("WARNING: bf16 not supported; falling back to float16")
            dtype = torch.float16
        else:
            dtype = torch.bfloat16
    else:
        dtype = {"float32": torch.float32, "float16": torch.float16}[args.dtype]

    device_type = "cuda" if "cuda" in device else "cpu"
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=dtype)

    # Set seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Load model & checkpoint
    model, checkpoint = load_model_from_checkpoint(args.out_dir, device, args.compile)

    # Initialize tokenizer (encode/decode functions)
    encode, decode = setup_tokenizer(checkpoint)

    # Prepare input prompt
    prompt = args.start
    if prompt.startswith("FILE:"):
        path = prompt[5:]
        with open(path, "r", encoding="utf-8") as f:
            prompt = f.read()
    start_ids = encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, :]

    # Generate samples
    with torch.no_grad():
        with ctx:
            for i in range(args.num_samples):
                out = model.generate(
                    x,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                )
                generated = out[0].tolist()
                print(decode(generated))
                print("----------")


if __name__ == "__main__":
    main()
