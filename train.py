import os
import time
import math
import pickle
import argparse
from contextlib import nullcontext
import wandb
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from config import TrainingConfig 

def parse_args():
    parser = argparse.ArgumentParser(description="Train a GPT‐style model")
    parser.add_argument(
        "--config_file", 
        type=str, 
        default="config/default.yaml",
        help="Path to YAML file containing all default hyperparameters"
    )
    # Instead of listing every single hyperparameter manually, we can:
    # 1) parse known argument keys via `--key value`
    # 2) collect any unknown args in a dict for overriding.
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Overrides for YAML keys in the form key=val (e.g. batch_size=24)"
    )
    return parser.parse_args()

def parse_override_args(override_list):
    """
    Given a list like ["batch_size=24", "learning_rate=1e-4"], 
    parse into { "batch_size": 24, "learning_rate": 0.0001 } 
    with proper types (int, float, bool, str).
    """
    parsed = {}
    for item in override_list:
        if "=" not in item:
            raise ValueError(f"Override '{item}' is not in key=val format")
        key, val_str = item.split("=", 1)
        # We’ll do a “best‐effort” literal eval to respect ints, floats, bools
        try:
            val = eval(val_str)  # safe as long as user controls these args
        except Exception:
            val = val_str
        parsed[key] = val
    return parsed

def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    
    path = os.path.join(data_dir, f"{split}.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    # if using CUDA, pin and move asynchronously
    if device.startswith("cuda"):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    """
    Run a quick estimate of train/val loss over eval_iters batches.
    Returns a dict: { "train": avg_train_loss, "val": avg_val_loss }.
    """
    losses = {}
    model.eval()
    for split in ["train", "val"]:
        tmp_losses = torch.zeros(eval_iters, device="cpu")
        for k in range(eval_iters):
            X, Y = get_batch_fn(split)
            with ctx:
                logits, loss = model(X, Y)
            tmp_losses[k] = loss.item()
        losses[split] = tmp_losses.mean().item()
    model.train()
    return losses

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

if __name__ == "__main__":
    args = parse_args()

    # 1) Load defaults from YAML file
    config = TrainingConfig.from_yaml(args.config_file)

    # 2) Parse and apply overrides (if any)
    overrides = parse_override_args(args.overrides)
    if overrides:
        config.update_from_dict(overrides)
        
    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get('RANK', -1)) != -1 # check if we are in DDP mode
    if ddp:
        init_process_group(backend=config.backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert config.gradient_accumulation_steps % ddp_world_size == 0
        config.gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        device   = config.device
        
    
    tokens_per_iter = config.gradient_accumulation_steps * ddp_world_size * config.batch_size * config.block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    
    # set the random seed
    if master_process:
        os.makedirs(config.out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Prepare dataset path
    data_dir = os.path.join('data', config.dataset)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Expected data directory: {data_dir}")
    
    # defining parameters
    block_size = config.block_size
    batch_size = config.batch_size
    eval_iters = config.eval_iters
    dtype      = config.dtype
    warmup_iters = config.warmup_iters
    learning_rate = config.learning_rate
    min_lr = config.min_lr
    lr_decay_iters = config.lr_decay_iters
    eval_only = config.eval_only
    
    
    # Try reading meta.pkl to get vocab_size
    meta_path = os.path.join(data_dir, "meta.pkl")
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta_vocab_size = meta.get("vocab_size", None)
        if meta_vocab_size is not None and master_process:
            print(f"Found vocab_size = {meta_vocab_size} (from {meta_path})")

    # Build model_args dict based on either “scratch” / “resume” / “gpt2*”
    model_args = dict(
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        block_size=config.block_size,
        bias=config.bias,
        vocab_size=None,  # fill in below
        dropout=config.dropout,
    )
    
    if config.init_from == "scratch":
        if master_process:
            print("Initializing a new model from scratch")
        if meta_vocab_size is None:
            # Default to GPT-2's vocab size (rounded to 50304)
            if master_process:
                print("No meta.pkl found—defaulting vocab_size to 50304")
            model_args["vocab_size"] = 50304
        else:
            model_args["vocab_size"] = meta_vocab_size
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

    elif config.init_from == "resume":
        if master_process:
            print(f"Resuming training from {config.out_dir}")
        ckpt_path = os.path.join(config.out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint["model_args"]
        # Force these fields to match
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = checkpoint_model_args[k]
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        # Strip off any unwanted prefix (if it exists)
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
        # We’ll re‐load the optimizer below from checkpoint
    elif config.init_from.startswith("gpt2"):
        if master_process:
            print(f"Initializing from OpenAI GPT‑2 weights: {config.init_from}")
        override_args = dict(dropout=config.dropout)
        model = GPT.from_pretrained(config.init_from, override_args)
        # Read off config created by HuggingFace
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = getattr(model.config, k)
    else:
        raise ValueError(f"Unexpected init_from: {config.init_from!r}")

    # Possibly crop block_size if user requested smaller context
    if config.block_size < model.config.block_size:
        model.crop_block_size(config.block_size)
        model_args["block_size"] = config.block_size

    # Move model to device
    model = model.to(device)

    # Initialize GradScaler (fp16 only)
    scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == "float16"))

    # setup optimizer (AdamW) and reload if “resume”
    optimizer = model.configure_optimizers(
        config.weight_decay,
        config.learning_rate,
        (config.beta1, config.beta2),
        device_type,
    )
    if config.init_from == "resume":
        optimizer.load_state_dict(checkpoint["optimizer"])
        del checkpoint  # free up memory

    # compile with PyTorch 2.0
    if config.compile and device_type == "cuda":
        if master_process:
            print("Compiling the model with torch.compile() (this may take ~1 min)...")
        model = torch.compile(model)

    # Wrap in DDP if needed
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # Define a local reference for “raw_model” (unwrapped) and training‐state
    raw_model = model.module if ddp else model
    if config.init_from != "resume":
        iter_num = 0
        best_val_loss = float("inf")

    # Set up data loader function
    get_batch_fn = lambda split: get_batch(split)

    if config.wandb_log and master_process:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=vars(config),
        )
        
    # Training loop
    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0

    while True:
        # Set learning rate for this iteration
        lr = get_lr(iter_num) if config.decay_lr else config.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Evaluate & checkpoint every eval_interval (only master)
        if iter_num % config.eval_interval == 0 and master_process:
            losses = estimate_loss()
            print(
                f"Step {iter_num}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )
            if config.wandb_log:
                wandb.log(
                    {
                        "iter": iter_num,
                        "train/loss": losses["train"],
                        "val/loss": losses["val"],
                        "lr": lr,
                        "mfu": running_mfu * 100,
                    }
                )
            if losses["val"] < best_val_loss or config.always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint_data = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": vars(config),
                    }
                    print(f"Saving checkpoint to {config.out_dir} …")
                    torch.save(
                        checkpoint_data, os.path.join(config.out_dir, "ckpt.pt")
                    )

        if iter_num == 0 and config.eval_only:
            break

        # Fetch first batch (only once, then reuse inside loop)
        if iter_num == 0:
            X, Y = get_batch_fn("train")

        # Forward/backward with gradient accumulation
        for micro_step in range(config.gradient_accumulation_steps):
            if ddp:
                # only sync gradients in the last micro step
                model.require_backward_grad_sync = (
                    micro_step == config.gradient_accumulation_steps - 1
                )
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / config.gradient_accumulation_steps

            # prefetch next batch asynchronously
            X, Y = get_batch_fn("train")
            scaler.scale(loss).backward()

        # Clip gradients (if requested)
        if config.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        # Step optimizer & scaler
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Logging (only master)
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % config.log_interval == 0 and master_process:
            lossf = loss.item() * config.gradient_accumulation_steps
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(
                    config.batch_size * config.gradient_accumulation_steps, dt
                )
                running_mfu = mfu if running_mfu < 0 else 0.9 * running_mfu + 0.1 * mfu
            print(
                f"Iter {iter_num}: loss {lossf:.4f}, "
                f"time {(dt*1000):.2f}ms, mfu {running_mfu*100:.2f}%"
            )

        iter_num += 1
        local_iter_num += 1

        # Termination
        if iter_num > config.max_iters:
            break

    # Clean up DDP
    if ddp:
        destroy_process_group()

    