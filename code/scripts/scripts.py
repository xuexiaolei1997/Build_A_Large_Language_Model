import numpy as np
import torch


def text_to_token_ids(text, tokenizer):
    """
    Args:
        text (str): Text to encode.
        tokenizer (Tokenizer): Tokenizer object used to encode text.
    Returns:
        torch.Tensor: Encoded text.
    """
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """
    Args:
        token_ids (torch.Tensor): Encoded text.
        tokenizer (Tokenizer): Tokenizer object used to encode text.
    Returns:
        str: Decoded text.
    """
    decoded = tokenizer.decode(token_ids.squeeze(0).tolist())
    return decoded


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """Generate text starting from the given index."""
    for _ in range(max_new_tokens):
        # >> context
        idx_cond = idx if idx.size(0) <= context_size else idx[-context_size:]

        # >> logits
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]  # last

        # >> topk
        if top_k is not None:
            top_logits, top_pos = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]  # min value
            # mask logits
            logits = torch.where(condition=logits < min_val, input=torch.tensor(float('-inf')), other=logits)
        
        # >> temperature
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def load_weights_into_gpt(gpt, model):

    def assign(left, right):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
        with torch.no_grad():
            left.copy_(torch.tensor(right))

    assign(gpt.tok_emb.weight, model.wte.weight)
    assign(gpt.pos_emb.weight, model.wpe.weight)

    for b in range(len(model.h)):
        q_w, k_w, v_w = np.split(
            model.h[b].attn.c_attn.weight, 3, axis=-1
        )

        assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        assign(gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            model.h[b].attn.c_attn.bias, 3, axis=-1
        )

        assign(gpt.trf_blocks[b].att.W_query.bias, q_b)
        assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
        assign(gpt.trf_blocks[b].att.W_value.bias, v_b)
        
        assign(gpt.trf_blocks[b].att.out_proj.weight, model.h[b].attn.c_proj.weight.T)
        assign(gpt.trf_blocks[b].att.out_proj.bias, model.h[b].attn.c_proj.bias)

        assign(gpt.trf_blocks[b].ff.layer[0].weight, model.h[b].mlp.c_fc.weight.T)
        assign(gpt.trf_blocks[b].ff.layer[0].bias, model.h[b].mlp.c_fc.bias)
        assign(gpt.trf_blocks[b].ff.layer[2].weight, model.h[b].mlp.c_proj.weight.T)
        assign(gpt.trf_blocks[b].ff.layer[2].bias, model.h[b].mlp.c_proj.bias)

        assign(gpt.trf_blocks[b].norm1.scale, model.h[b].ln_1.weight)
        assign(gpt.trf_blocks[b].norm1.shift, model.h[b].ln_1.bias)
        assign(gpt.trf_blocks[b].norm2.scale, model.h[b].ln_2.weight)
        assign(gpt.trf_blocks[b].norm2.shift, model.h[b].ln_2.bias)
    
    assign(gpt.final_norm.scale, model.ln_f.weight)
    assign(gpt.final_norm.shift, model.ln_f.bias)
    assign(gpt.out_head.weight, model.wte.weight)

