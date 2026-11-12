"""Ask the same VLM "Is there a {object} in the image?" and read P(yes) from its first answer token."""
import torch

from ..vlm.model import encode, image_messages, reset_rope

QUESTION = "Is there a {obj} in the image? Answer yes or no."


def _first_ids(tok, words):
    return sorted({tok.encode(w, add_special_tokens=False)[0] for w in words})


@torch.no_grad()
def p_yes(model, processor, image_path, categories, cfg):
    """{category: P(yes)} using one batched forward pass over all questions for this image."""
    if not categories:
        return {}
    tok = processor.tokenizer
    tok.padding_side = "left"
    yes_ids = _first_ids(tok, ["Yes", "yes"])
    no_ids = _first_ids(tok, ["No", "no"])
    msgs = [image_messages(image_path, QUESTION.format(obj=c), cfg) for c in categories]
    batch = encode(processor, msgs, model.device)
    reset_rope(model)
    logits = model(**batch, cache_position=torch.arange(batch["input_ids"].shape[1], device=model.device)).logits[:, -1].float()
    probs = torch.softmax(logits, dim=-1)
    py = probs[:, yes_ids].sum(-1)
    pn = probs[:, no_ids].sum(-1)
    return dict(zip(categories, (py / (py + pn)).cpu().tolist()))
