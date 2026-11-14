"""Run a frozen local VLM and extract relative Yes/No scores on one GPU."""
import argparse
import json
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Set cache location before importing Hugging Face libraries.
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))
# Windows supports ordinary cached files without enabling developer mode.
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

import torch
from PIL import Image, ImageDraw, ImageOps
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

MODEL_ID = 'Qwen/Qwen2.5-VL-3B-Instruct'
DEFAULT_STATEMENT = 'The red square is to the left of the blue circle.'


def demo_image():
    """A controlled input for installation checks; not a research benchmark."""
    image = Image.new('RGB', (448, 280), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((45, 90, 145, 190), fill='red')
    draw.ellipse((300, 90, 400, 190), fill='blue')
    return image


def score(model, processor, image, statement, hidden_path=None):
    prompt = f'Statement: "{statement}"\nIs this statement true in the image? Answer Yes or No.'
    messages = [{'role': 'user', 'content': [
        {'type': 'image'}, {'type': 'text', 'text': prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Validate tokens in the actual answer context; fail rather than score partial words.
    prefix_ids = processor.tokenizer.encode(text, add_special_tokens=False)
    answer_ids = []
    for answer in ('Yes', 'No'):
        joined = processor.tokenizer.encode(text + answer, add_special_tokens=False)
        suffix = joined[len(prefix_ids):]
        if joined[:len(prefix_ids)] != prefix_ids or len(suffix) != 1:
            raise RuntimeError(f'{answer!r} is not one token in this context; sequence scoring is required.')
        answer_ids.append(suffix[0])

    inputs = processor(text=[text], images=[image], return_tensors='pt').to('cuda:0')
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model(
            **inputs, use_cache=False, output_hidden_states=hidden_path is not None,
            return_dict=True, logits_to_keep=1,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    logits = output.logits[0, -1].float()
    if not torch.isfinite(logits).all().item():
        raise RuntimeError('Non-finite logits: do not use this run for experiments.')
    pair = torch.softmax(logits[answer_ids], dim=0)
    log_probs = torch.log_softmax(logits, dim=0)
    choice_mass = torch.exp(torch.logsumexp(log_probs[answer_ids], dim=0)).item()
    top = torch.topk(log_probs, 5)
    result = {
        'statement': statement,
        'p_yes': pair[0].item(), 'p_no': pair[1].item(),
        'binary_answer': 'Yes' if pair[0].item() >= 0.5 else 'No',
        'answer_confidence': pair.max().item(),
        'answer_token_ids': dict(zip(('Yes', 'No'), answer_ids)),
        'yes_no_full_vocabulary_mass': choice_mass,
        'unrestricted_top_tokens': [
            {'text': processor.tokenizer.decode([idx]), 'probability': prob.exp().item()}
            for idx, prob in zip(top.indices.tolist(), top.values)
        ],
        'input_tokens': inputs['input_ids'].shape[1],
        'image_grid_thw': inputs['image_grid_thw'].tolist(),
        'forward_seconds': round(elapsed, 3),
        'peak_allocated_vram_gib': round(torch.cuda.max_memory_allocated() / 2**30, 3),
    }
    if hidden_path is not None:
        layer_count = len(output.hidden_states) - 1
        layers = sorted({max(1, layer_count // 4), max(1, layer_count // 2), layer_count})
        torch.save({
            'position': 'final prompt token, before answering',
            'hidden_state_indices': layers,
            'vectors': {idx: output.hidden_states[idx][0, -1].detach().cpu() for idx in layers},
        }, hidden_path)
        result['hidden_states_file'] = str(hidden_path)
    return result


def load_vlm(revision='main', max_pixels=256 * 28 * 28):
    """Shared frozen-model configuration for the demo and dataset pilot."""
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Run scripts/check_gpu.py first.')
    torch.manual_seed(42)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f'Loading {MODEL_ID} in 4-bit NF4 on {torch.cuda.get_device_name(0)}...', flush=True)
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=revision, min_pixels=4 * 28 * 28,
        max_pixels=max_pixels, use_fast=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=revision, device_map={'': 0}, dtype=dtype,
        attn_implementation='sdpa',
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
            llm_int8_skip_modules=['visual', 'lm_head'],
        ),
    ).eval()
    model.requires_grad_(False)
    metadata = {
        'model_id': MODEL_ID, 'model_revision': getattr(model.config, '_commit_hash', None),
        'quantization': '4-bit NF4 double quantization; visual and lm_head excluded',
        'compute_dtype': str(dtype), 'attention': 'sdpa', 'batch_size': 1,
        'max_pixels': max_pixels, 'gpu': torch.cuda.get_device_name(0),
        'processor_use_fast': True, 'min_pixels': 4 * 28 * 28,
        'python': platform.python_version(), 'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda,
        'packages': {p: version(p) for p in ('transformers', 'accelerate', 'bitsandbytes', 'huggingface-hub')},
        'model_load_seconds': round(time.perf_counter() - started, 3),
    }
    return model, processor, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, help='Local image; omit to use the controlled shapes demo.')
    parser.add_argument('--statement', default=DEFAULT_STATEMENT)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'smoke_test.json')
    parser.add_argument('--revision', default='main', help='Use a saved commit hash for reproducible reruns.')
    parser.add_argument('--max-pixels', type=int, default=256 * 28 * 28)
    parser.add_argument('--save-hidden', action='store_true')
    parser.add_argument('--suite', action='store_true', help='Run original, mirrored-equivalent, opposite, and grey-image checks on the demo.')
    args = parser.parse_args()
    if args.suite and (args.image is not None or args.statement != DEFAULT_STATEMENT):
        parser.error('--suite uses the default controlled demo and statement.')
    if args.max_pixels < 4 * 28 * 28:
        parser.error('--max-pixels must allow at least four visual tokens.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model, processor, metadata = load_vlm(args.revision, args.max_pixels)
    image = ImageOps.exif_transpose(Image.open(args.image)).convert('RGB') if args.image else demo_image()
    if args.image is None:
        image.save(args.output.parent / 'demo_shapes.png')
    cases = [('original', image, args.statement, True if args.image is None and args.statement == DEFAULT_STATEMENT else None)]
    if args.suite:
        opposite = 'The red square is to the right of the blue circle.'
        cases.extend([
            ('opposite', image, opposite, False),
            ('mirrored_equivalent', ImageOps.mirror(image), opposite, True),
            ('blind', Image.new('RGB', image.size, (128, 128, 128)), args.statement, None),
        ])
    results = []
    for name, case_image, statement, expected in cases:
        hidden_path = args.output.with_suffix('.hidden.pt') if args.save_hidden and name == 'original' else None
        result = score(model, processor, case_image, statement, hidden_path)
        result.update(case=name, expected_truth=expected)
        results.append(result)
        print(f'{name}: {result["binary_answer"]}, P(Yes)={result["p_yes"]:.4f}, '
              f'{result["forward_seconds"]} s, peak={result["peak_allocated_vram_gib"]} GiB', flush=True)
    report = {
        **metadata,
        'score_definition': 'softmax of canonical Yes and No token logits only; not calibrated correctness probability',
        'results': results,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
