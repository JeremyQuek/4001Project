"""Frozen VLM loader and Yes/No scorer.

The model is never trained. The group's VSR experiment loads and scores the
model in the same way (same revision, 4-bit NF4 quantization, pixel budget,
prompt, answer-token check and hidden-state definition), so both experiments
use an identical base model.

Run this file directly for the one-image smoke test:
    python scripts/vlm.py
"""
import sys
import time
from importlib.metadata import version

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from common import OUTPUTS, PROMPT_TEMPLATE, VLM, write_json


def load_vlm():
    """Load the pinned Qwen2.5-VL revision in 4-bit NF4 on the single GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable.')
    torch.manual_seed(42)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('bfloat16 is required to match the frozen configuration.')
    dtype = torch.bfloat16
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        VLM['model_id'], revision=VLM['model_revision'], min_pixels=VLM['min_pixels'],
        max_pixels=VLM['max_pixels'], use_fast=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM['model_id'], revision=VLM['model_revision'], device_map={'': 0}, dtype=dtype,
        attn_implementation='sdpa',
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype, llm_int8_skip_modules=['visual', 'lm_head']),
    ).eval()
    model.requires_grad_(False)
    metadata = {
        **VLM, 'loaded_revision': getattr(model.config, '_commit_hash', None),
        'gpu': torch.cuda.get_device_name(0), 'python': sys.version.split()[0],
        'torch': torch.__version__,
        'packages': {p: version(p) for p in ('transformers', 'accelerate', 'bitsandbytes')},
        'model_load_seconds': round(time.perf_counter() - started, 1),
    }
    if metadata['loaded_revision'] != VLM['model_revision']:
        raise RuntimeError('Loaded model revision differs from the frozen revision.')
    return model, processor, metadata


def load_image(path):
    """Open an image, apply its EXIF orientation and convert it to RGB."""
    return ImageOps.exif_transpose(Image.open(path)).convert('RGB')


def score(model, processor, image, statement):
    """One forward pass: relative Yes/No support and hidden states at the final prompt token."""
    prompt = PROMPT_TEMPLATE.format(statement=statement)
    messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # "Yes" and "No" must each be exactly one token in this exact answer context.
    prefix_ids = processor.tokenizer.encode(text, add_special_tokens=False)
    answer_ids = []
    for answer in ('Yes', 'No'):
        joined = processor.tokenizer.encode(text + answer, add_special_tokens=False)
        suffix = joined[len(prefix_ids):]
        if joined[:len(prefix_ids)] != prefix_ids or len(suffix) != 1:
            raise RuntimeError(f'{answer!r} is not a single token in this prompt context.')
        answer_ids.append(suffix[0])

    inputs = processor(text=[text], images=[image], return_tensors='pt').to('cuda:0')
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model(**inputs, use_cache=False, output_hidden_states=True,
                       return_dict=True, logits_to_keep=1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    logits = output.logits[0, -1].float()
    if not torch.isfinite(logits).all().item():
        raise RuntimeError('Non-finite logits.')
    # Relative Yes/No support: softmax over the Yes and No logits only (ASSUMPTIONS.md, A4).
    pair = torch.softmax(logits[answer_ids], dim=0)
    log_probs = torch.log_softmax(logits, dim=0)
    # hidden_states[0] is the embedding output, so index L is decoder layer L (ASSUMPTIONS.md, A7).
    if len(output.hidden_states) - 1 != 36:
        raise RuntimeError('Unexpected number of decoder layers.')
    hidden = np.stack([output.hidden_states[layer][0, -1].float().cpu().numpy()
                       for layer in VLM['hidden_layers']]).astype(np.float32)
    if hidden.shape != (3, 2048) or not np.isfinite(hidden).all():
        raise RuntimeError('Invalid hidden states.')
    return {
        'p_yes': pair[0].item(),
        'logit_yes': logits[answer_ids[0]].item(), 'logit_no': logits[answer_ids[1]].item(),
        # Share of the full-vocabulary probability that falls on Yes/No (sanity check).
        'yes_no_mass': torch.exp(torch.logsumexp(log_probs[answer_ids], dim=0)).item(),
        'top_token': processor.tokenizer.decode([int(logits.argmax())]),
        'answer_token_ids': answer_ids, 'input_tokens': int(inputs['input_ids'].shape[1]),
        'forward_seconds': elapsed, 'hidden': hidden,
    }


def main():
    """Smoke test on one controlled image (red square left of blue circle)."""
    image = Image.new('RGB', (448, 280), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((45, 90, 145, 190), fill='red')
    draw.ellipse((300, 90, 400, 190), fill='blue')
    model, processor, metadata = load_vlm()
    results = []
    for statement, expected in [('The red square is to the left of the blue circle.', 1),
                                ('The red square is to the right of the blue circle.', 0)]:
        result = score(model, processor, image, statement)
        result.pop('hidden')
        result.update(statement=statement, expected_truth=expected)
        results.append(result)
        print(f'{statement} P(Yes)={result["p_yes"]:.4f} top={result["top_token"]!r} '
              f'{result["forward_seconds"]:.2f}s', flush=True)
    metadata['peak_allocated_vram_gib'] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    write_json(OUTPUTS / 'smoke_test.json', {**metadata, 'results': results})
    print('Peak VRAM GiB:', metadata['peak_allocated_vram_gib'], '| load s:', metadata['model_load_seconds'])


if __name__ == '__main__':
    main()
