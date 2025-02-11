import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    decoded = tokenizer.decode(token_ids.squeeze(0).tolist())
    return decoded


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=-1)
    return idx


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(len(data_loader), num_batches)
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches


def evaluete_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    excoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model, excoded, 50 , context_size)
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(f">> {start_context} --> {decoded_text}")
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_token_seen = [], [], []
    token_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            token_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluete_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_token_seen.append(token_seen)
                print(f">> Epoch {epoch + 1}, step {global_step: 06d}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
    
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_token_seen


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
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



def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]  # Logits of last output token·
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples


def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # Create a second x-axis for examples seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(examples_seen, train_values, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.show()


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(epochs_seen, train_losses, label="train")
    ax.plot(epochs_seen, val_losses, linestyle='-.', label="val")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()

    ax1 = ax.twiny()
    ax1.plot(tokens_seen, train_losses, alpha=0)
    ax1.set_xlabel("Tokens Seen")

    fig.tight_layout()
    plt.show()


def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"):
    batch_max_length = max(len(entry)+1 for entry in batch)

    inputs_lst, targets_lst = [], []

    for entry in batch:
        new_entry = entry.copy()
        new_entry += [pad_token_id]

        padded = new_entry + [pad_token_id] * (batch_max_length - len(new_entry))

        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor
