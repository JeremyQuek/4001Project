"""Caption generation (pass 1) and the teacher-forced activation pass (pass 2).

Pass 1 only produces token ids. Pass 2 re-runs prompt + those exact ids in one
forward pass with output_hidden_states=True, which gives aligned hidden states for
every caption position without per-step generate() bookkeeping or padding effects.
"""
import torch

from .model import encode, image_messages, reset_rope, stop_token_ids
from .tokens import (MAX_OFFSET, POSITION_OFFSETS, char_to_token, mention_token_spans,  # noqa: F401
                     strip_after_stop, token_char_ends)


@torch.no_grad()
def generate_captions(model, processor, image_paths, prompt, max_new_tokens, cfg):
    """Greedy captions for a batch of images. Returns [(token_ids, caption_text)]."""
    processor.tokenizer.padding_side = "left"
    msgs = [image_messages(p, prompt, cfg) for p in image_paths]
    batch = encode(processor, msgs, model.device)
    reset_rope(model)
    out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[:, batch["input_ids"].shape[1]:].tolist()
    stops = stop_token_ids(model, processor)
    results = []
    for ids in gen:
        ids = strip_after_stop(ids, stops)
        results.append((ids, processor.tokenizer.decode(ids, skip_special_tokens=True)))
    return results


@torch.no_grad()
def teacher_forced(model, processor, image_path, prompt, gen_ids, layers, cfg, want_attention=False):
    """Run prompt + caption ids once. Returns a dict with:

    hidden:   float16 tensor [n_layers, MAX_OFFSET + G, D]. Index MAX_OFFSET + i is caption token i,
              so index MAX_OFFSET + i - k is the state k positions before it (may lie in the prompt).
    prob:     [G] probability the model gave to each caption token when it chose it.
    entropy:  [G] entropy of the next-token distribution at that choice.
    att_image:[n_layers, G] share of attention (mean over heads) on image tokens at the choosing step,
              only if want_attention (needs attn_implementation="eager").
    """
    processor.tokenizer.padding_side = "left"
    batch = encode(processor, [image_messages(image_path, prompt, cfg)], model.device)
    prompt_ids = batch["input_ids"]
    P = prompt_ids.shape[1]
    G = len(gen_ids)
    caption = torch.tensor([gen_ids], device=model.device, dtype=prompt_ids.dtype)
    input_ids = torch.cat([prompt_ids, caption], dim=1)
    reset_rope(model)
    out = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        pixel_values=batch["pixel_values"],
        image_grid_thw=batch["image_grid_thw"],
        cache_position=torch.arange(input_ids.shape[1], device=model.device),
        output_hidden_states=True,
        output_attentions=want_attention,
        use_cache=False,
    )
    # hidden_states[0] is the embedding output; hidden_states[l] is the output of layer l.
    lo = P - MAX_OFFSET
    hidden = torch.stack([out.hidden_states[l][0, lo:P + G] for l in layers]).to(torch.float16).cpu()

    # logits[j] predicts token j+1, so caption token i (absolute P+i) was chosen from logits[P+i-1].
    logits = out.logits[0, P - 1:P + G - 1].float()
    logp = torch.log_softmax(logits, dim=-1)
    target = caption[0].unsqueeze(1)
    prob = logp.gather(1, target).squeeze(1).exp().cpu()
    entropy = -(logp.exp() * logp).sum(-1).cpu()

    result = {"hidden": hidden, "prob": prob, "entropy": entropy, "prompt_len": P}
    if want_attention:
        is_image = (input_ids[0] == model.config.image_token_id)
        rows = []
        for l in layers:
            # attentions is indexed by layer starting at 0 = first decoder layer, so layer l -> l-1.
            att = out.attentions[l - 1][0, :, P - 1:P + G - 1, :].float()  # [heads, G, S]
            rows.append((att[..., is_image].sum(-1) / att.sum(-1)).mean(0).cpu())
        result["att_image"] = torch.stack(rows)
    return result
