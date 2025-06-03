import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import time, math, os
import inspect

@dataclass
class GPTConfig:
    vocab_size: int = 50257 # number of tokens in the vocabulary
    block_size: int = 1024 # maximum context length
    n_embd: int = 768 # embedding size (dimension of the token embeddings)
    n_head: int = 12 # number of attention heads in the multi-head attention layer
    n_layer: int = 12 # number of transformer blocks in the model
    dropout: float = 0.0 # dropout rate for regularization
    bias: bool = True # bias in Linears
    
class CausalSelfAttention(nn.Module):
    """ Self-attention layer """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding size must be divisible by number of heads"
        self.c_attn  = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)  # linear layer for query, key, value
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)  # linear layer for output projection
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        #flash attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')  # check if scaled dot product attention is available
        if not self.flash:
            print("Warning: Flash attention is not available, using standard attention instead.")
            # not really a 'bias', but a mask for causal attention
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))  # causal mask for attention

    def forward(self, x):
        B, T, C = x.size() # (batch size, sequence length, embedding size (n_embd))
        # compute query, key, value matrices
        qkv = self.c_attn(x)  # (B, T, 3 * n_embd)
        q, k, v = qkv.split(self.n_embd, dim=2)  # split into query, key, value (each of shape (B, T, n_embd))
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # compute attention scores
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            attn = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)  # (B, nh, T, hs) @ (B, nh, hs, T) --> (B, nh, T, T)
            attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))  # apply causal mask
            attn = F.softmax(attn, dim=-1)  # apply softmax to get attention weights
            attn = self.attn_dropout(attn)  # apply dropout to attention weights
            y = attn @ v  # (B, nh, T, T) @ (B, nh, T, hs) --> (B, nh, T, hs)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, nh, T, hs) --> (B, T, n_embd)
        y = self.resid_dropout(self.c_proj(y))  # (B, T, n_embd) --> (B, T, n_embd)
        return y  # output of self-attention layer (B, T, n_embd)

class MLP(nn.Module):
    """ Feedforward network with two linear layers and GELU activation """
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)  # first linear layer
        self.gelu = nn.GELU(approximate="tanh")  # GELU activation function
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)  # second linear layer
        self.dropout = nn.Dropout(config.dropout)  # dropout layer for regularization
                
    def forward(self, x):
        x = self.c_fc(x) # (B, T, n_embd) --> (B, T, 4 * n_embd)
        x = self.gelu(x) # apply GELU activation
        x = self.c_proj(x) # (B, T, 4 * n_embd) --> (B, T, n_embd)
        x = self.dropout(x)  # apply dropout
        return x # (B, T, n_embd)
    
class Block(nn.Module):
    """ Transformer block consisting of self-attention and feedforward layers """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)  # layer normalization
        self.attn = CausalSelfAttention(config) # self-attention layer
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)  # feedforward network
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    """ GPT2 model with transformer architecture """
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size > 0, "Vocabulary size must be greater than 0"
        assert config.block_size > 0, "Block size must be greater than 0"
        assert config.n_embd > 0, "Embedding size must be greater than 0"
        
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # word token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),  # positional encoding
            drop = nn.Dropout(config.dropout), # dropout layer for regularization
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # transformer blocks
            ln_f = nn.LayerNorm(config.n_embd),  # final layer normalization  
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # linear layer for language modeling head
        
        # weight sharing scheme: share the token embedding with the language modeling head
        self.transformer.wte.weight = self.lm_head.weight
        
        #initialize weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    def forward(self, idx, targets=None):
        device = idx.device # idx --> (B, T) where B is batch size and T is sequence length
        B, T = idx.size()   # get batch size and sequence length
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        pos = torch.arange(0, T, dtype = torch.long , device=device)  # positional indices
        pos_emb = self.transformer.wpe(pos)  # positional embeddings
        tok_emb = self.transformer.wte(idx)  # token embeddings
        x = self.transformer.drop(tok_emb + pos_emb)  # combine token and positional embeddings
        for block in self.transformer.h:  # pass through each transformer block
            x = block(x)  # forward pass through the block
        x = self.transformer.ln_f(x)  # final layer normalization
        
        loss = None
        if targets is not None:
            logits = self.lm_head(x)  # (B, T, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)  # compute loss if targets are provided
        
        return logits, loss
    
    # referenced below methods from https://github.com/karpathy/nanoGPT
    def crop_block_size(self, block_size):
        # Adjust the model to use a smaller block size if required.
        # For example, if we load a GPT-2 pretrained model with a block size of 1024,
        # but need a smaller block size for a simpler or more efficient model.
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
        
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer
    
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate a sequence of tokens by autoregressively sampling from the model's output.

        Args:
            idx (torch.LongTensor): Conditioning sequence of indices with shape (batch_size, sequence_length).
            max_new_tokens (int): Number of new tokens to generate.
            temperature (float): Sampling temperature to control randomness (default: 1.0).
            top_k (int, optional): If specified, only consider the top_k logits for sampling.

        Returns:
            torch.LongTensor: Generated sequence of indices with shape (batch_size, sequence_length + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            # crop context if it exceeds block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond) # get logits for the current sequence
            logits = logits[:, -1, :] / temperature # scale logits by temperature
            # keep only top_k logits if specified
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # compute probabilities
            probs = F.softmax(logits, dim=-1)
            # sample next token using multinomial sampling
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1) # append the sampled token to the sequence
            
        return idx
