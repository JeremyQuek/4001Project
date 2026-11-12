"""Loading Qwen2-VL and turning (image, prompt) into model inputs."""
import torch


def load_vlm(cfg, attn_implementation=None):
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    dtype = getattr(torch, cfg.model.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg.model.name,
        torch_dtype=dtype,
        attn_implementation=attn_implementation or cfg.model.attn_implementation,
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(cfg.model.name)
    return model, processor


def stop_token_ids(model, processor):
    ids = model.generation_config.eos_token_id
    ids = set(ids if isinstance(ids, (list, tuple)) else [ids])
    ids.add(processor.tokenizer.eos_token_id)
    if processor.tokenizer.pad_token_id is not None:
        ids.add(processor.tokenizer.pad_token_id)
    return ids


def image_messages(image_path, text, cfg):
    """One user turn: the image followed by the text prompt.

    min/max_pixels go on the image entry because qwen_vl_utils resizes the image there."""
    image = {"type": "image", "image": str(image_path)}
    if cfg.model.get("min_pixels"):
        image["min_pixels"] = cfg.model.min_pixels
    if cfg.model.get("max_pixels"):
        image["max_pixels"] = cfg.model.max_pixels
    return [{"role": "user", "content": [image, {"type": "text", "text": text}]}]


def reset_rope(model):
    """Qwen2-VL caches `rope_deltas` from the last prefill and reuses it whenever a forward call
    does not start at cache position 0. Clear it before every new prompt so a fresh image never
    inherits the previous image's 3D rotary positions."""
    for m in (model, getattr(model, "model", None)):
        if m is not None and hasattr(m, "rope_deltas"):
            m.rope_deltas = None


def encode(processor, messages_list, device):
    """Batch of chat messages -> model inputs (prompt only, ending with the assistant header)."""
    from qwen_vl_utils import process_vision_info

    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
    images, _ = process_vision_info(messages_list)
    batch = processor(text=texts, images=images, padding=True, return_tensors="pt")
    return batch.to(device)
