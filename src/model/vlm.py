"""Qwen2.5-VL-3B-Instruct wrapper: ask yes/no questions about images, get p_yes / p_no and hidden states.

One forward pass per question, no generation: the answer is read from the next-token scores of "Yes" and "No",
and the hidden states are taken at the last prompt token, the position that produces the answer.

Weights   src/model/models/Qwen2.5-VL-3B-Instruct/ (download() fetches them from Hugging Face, about 7 GB)

Stage 4 (VLM.run / run): answers every question in a queries file and writes outputs/answers/:
    {split}_records.json   one TrainRecord per line
    {split}_hidden.npy     float16 [N, n_saved_layers, d_model]; row i belongs to record i
    layers.json            which model layer each index of the second axis is
Work is saved in shards ({split}_shards/), so a rerun after a disconnect carries on where it stopped.
Photos are read from visual_genome/images/{image_id}.jpg, downloaded first from the URL in image_data.json if missing.

Library: vlm = VLM(); vlm.ask([image], ["Is the cup ...? Answer yes or no."]) -> [VLMAnswer]
         vlm.run(queries_path, split="train", limit=200)   one queries file
         run(splits=("train", "val", "test"), limit=None)   every split, as main.ipynb calls it
         draw_boxes(image, subj_bbox, obj_bbox) marks the subject in red and the object in blue.
Script:  python src/model/vlm.py --download
         python src/model/vlm.py --run [--splits train val test] [--limit N]
         python src/model/vlm.py --image photo.jpg --question "..." [--subj-box X Y W H --obj-box X Y W H]
"""

import argparse
import json
import sys
import time
import urllib.request
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lib import paths
from data.records import read_records, write_records
from data.schemas import Answer, BBox, Query, TrainRecord
from lib.device import pick_device

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# Images are resized so their area falls in this range; the cap keeps the number of image tokens (and the run
# time) bounded. 448 x 448 = about 256 image tokens, since each token covers 28 x 28 pixels.
MIN_PIXELS = 128 * 28 * 28
MAX_PIXELS = 448 * 448

RED = (255, 0, 0)
BLUE = (0, 0, 255)


def download(model_dir: Path | None = None) -> Path:
    from huggingface_hub import snapshot_download

    target = Path(model_dir or paths.QWEN_DIR)
    snapshot_download(MODEL_ID, local_dir=target)
    return target


@cache
def image_urls() -> dict[int, str]:
    """image_id -> photo URL, from image_data.json (zipped or not) in the Visual Genome folder."""
    for name in ("image_data.json", "image_data.json.zip"):
        path = paths.vg_dir() / name
        if path.exists():
            if path.suffix == ".zip":
                with zipfile.ZipFile(path) as z, z.open("image_data.json") as f:
                    data = json.load(f)
            else:
                data = json.loads(path.read_text())
            return {d["image_id"]: d["url"] for d in data}
    raise FileNotFoundError(f"image_data.json not found in {paths.vg_dir()}; it holds the photo URLs")


def load_image(image_id: int) -> Image.Image:
    """The photo for image_id, downloaded into visual_genome/images/ the first time."""
    path = paths.images_dir() / f"{image_id}.jpg"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(image_urls()[image_id], tmp)
        tmp.rename(path)
    with Image.open(path) as image:
        return image.convert("RGB")


def draw_boxes(image: Image.Image, subj_bbox: BBox, obj_bbox: BBox) -> Image.Image:
    """A copy of image with the subject boxed in red and the object in blue."""
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)
    width = max(2, round(min(out.size) / 150))
    for box, colour in ((subj_bbox, RED), (obj_bbox, BLUE)):
        draw.rectangle((box.x, box.y, box.x + box.w, box.y + box.h), outline=colour, width=width)
    return out


@dataclass
class VLMAnswer:
    p_yes: float  # P(yes) / (P(yes) + P(no)) at the answer step
    p_no: float
    hidden: np.ndarray  # float16 [n_saved_layers, d_model], at the last prompt token

    @property
    def answer(self) -> str:
        return "yes" if self.p_yes >= self.p_no else "no"


def default_layers(n_layers: int) -> list[int]:
    """Every 4th hidden state plus the last; hidden state 0 is the embedding layer."""
    layers = list(range(0, n_layers + 1, 4))
    if layers[-1] != n_layers:
        layers.append(n_layers)
    return layers


class VLM:
    def __init__(self, model_dir: Path | None = None, layers: Sequence[int] | None = None, device: str = "auto",
                 dtype: torch.dtype | None = None):
        model_dir = Path(model_dir or paths.QWEN_DIR)
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(f"no Qwen weights in {model_dir}; run `python src/model/vlm.py --download`")
        self.device = pick_device(device)
        # a T4 has no bfloat16, so float16 on CUDA; float32 elsewhere
        self.dtype = dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)

        self.processor = AutoProcessor.from_pretrained(model_dir)
        # set on the image processor itself: the saved config's min/max_pixels override a size passed to
        # from_pretrained. For Qwen, shortest_edge / longest_edge are the min / max image area in pixels.
        image_processor = self.processor.image_processor
        image_processor.min_pixels, image_processor.max_pixels = MIN_PIXELS, MAX_PIXELS
        image_processor.size = {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}
        # left padding keeps every question's last prompt token at position -1 in a batch
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir, dtype=self.dtype, attn_implementation="sdpa"
        ).to(self.device).eval()

        n_layers = self.model.config.text_config.num_hidden_layers
        self.layers = list(layers) if layers else default_layers(n_layers)
        self.yes_ids = self._first_token_ids(["Yes", "yes", " Yes", " yes"])
        self.no_ids = self._first_token_ids(["No", "no", " No", " no"])

    def _first_token_ids(self, words: list[str]) -> list[int]:
        tokenizer = self.processor.tokenizer
        return sorted({tokenizer.encode(w, add_special_tokens=False)[0] for w in words})

    def prompt(self, question: str, previous_answer: str | None = None, follow_up: str | None = None) -> str:
        """The chat prompt; with previous_answer and follow_up it is a second turn asking the question again."""
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        if previous_answer is not None:
            messages += [
                {"role": "assistant", "content": [{"type": "text", "text": previous_answer}]},
                {"role": "user", "content": [{"type": "text", "text": follow_up}]},
            ]
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @property
    def decoder_layers(self) -> torch.nn.ModuleList:
        return self.model.model.language_model.layers

    def hook_module(self, layer: int) -> torch.nn.Module:
        """The module whose output is hidden state `layer` (as numbered in hidden_states and self.layers).

        hidden_states[L] is the output of decoder layer L - 1, except the last, which is taken after the final norm.
        Layer 0 is the embeddings, which steering doesn't edit.
        """
        n_layers = len(self.decoder_layers)
        if not 1 <= layer <= n_layers:
            raise ValueError(f"layer must be between 1 and {n_layers}, got {layer}")
        if layer == n_layers:
            return self.model.model.language_model.norm
        return self.decoder_layers[layer - 1]

    @torch.no_grad()
    def ask(self, images: Sequence[Image.Image], questions: Sequence[str],
            previous_answers: Sequence[str] | None = None, follow_ups: Sequence[str] | None = None) -> list[VLMAnswer]:
        """One batched forward pass; images[i] goes with questions[i].

        With previous_answers and follow_ups, each question is asked again as a second turn: the model's earlier
        answer, then the follow-up message.
        """
        if previous_answers is None:
            prompts = [self.prompt(q) for q in questions]
        else:
            prompts = [self.prompt(q, a, f) for q, a, f in zip(questions, previous_answers, follow_ups)]
        inputs = self.processor(text=prompts, images=list(images), padding=True,
                                return_tensors="pt").to(self.device)
        out = self.model(**inputs, output_hidden_states=True)

        probs = torch.softmax(out.logits[:, -1, :].float(), dim=-1)
        p_yes = probs[:, self.yes_ids].sum(dim=-1)
        p_no = probs[:, self.no_ids].sum(dim=-1)
        total = p_yes + p_no
        hidden = torch.stack([out.hidden_states[layer][:, -1, :] for layer in self.layers], dim=1)
        hidden = hidden.to(torch.float16).cpu().numpy()

        return [VLMAnswer(p_yes=float(y), p_no=float(n), hidden=h)
                for y, n, h in zip((p_yes / total).tolist(), (p_no / total).tolist(), hidden)]

    def run(self, queries_path: Path, split: str, out_dir: Path | None = None, limit: int | None = None,
            batch_size: int = 8, shard_size: int = 512) -> Path:
        """Answer the first `limit` questions of a queries file (all if None); returns the records file.

        Questions are asked batch_size at a time; the four variants of a relation sit next to each other in the
        file and share one drawn image. Each shard of shard_size questions is saved as soon as it is done, and
        shards already on disk are skipped, so a rerun resumes.
        """
        out_dir = Path(out_dir or paths.answers_dir())
        queries = read_records(queries_path, Query)[:limit]
        shard_dir = out_dir / f"{split}_shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shards = []

        for start in range(0, len(queries), shard_size):
            chunk = queries[start:start + shard_size]
            stem = shard_dir / f"{start:07d}-{start + len(chunk):07d}"
            shards.append(stem)
            # the records file is written last, so its presence means the shard is complete
            if stem.with_suffix(".json").exists():
                continue

            began = time.time()
            records, hidden = [], []
            boxed = {}  # rel_id -> drawn image, shared by a relation's four variants
            for b in range(0, len(chunk), batch_size):
                batch = chunk[b:b + batch_size]
                for q in batch:
                    if q.rel_id not in boxed:
                        boxed[q.rel_id] = draw_boxes(load_image(q.image_id), q.subj_bbox, q.obj_bbox)
                answers = self.ask([boxed[q.rel_id] for q in batch], [q.query_text for q in batch])
                for q, a in zip(batch, answers):
                    model_answer = Answer(a.answer)
                    records.append(TrainRecord(**q.model_dump(), model_answer=model_answer, p_yes=a.p_yes,
                                               p_no=a.p_no, wrong=int(model_answer is not q.gt_answer)))
                    hidden.append(a.hidden)

            np.save(stem.with_suffix(".npy"), np.stack(hidden))
            write_records(stem.with_suffix(".json"), records)
            secs = time.time() - began
            print(f"{split}: {start + len(chunk)}/{len(queries)} questions  "
                  f"({len(chunk) / secs:.1f}/s, {sum(r.wrong for r in records)} wrong in this shard)")

        return self._merge(shards, out_dir, split)

    def _merge(self, shards: list[Path], out_dir: Path, split: str) -> Path:
        """Join the shards into {split}_records.json and {split}_hidden.npy, without loading every row at once."""
        records = [r for stem in shards for r in read_records(stem.with_suffix(".json"), TrainRecord)]
        first = np.load(shards[0].with_suffix(".npy"), mmap_mode="r")
        merged = np.lib.format.open_memmap(out_dir / f"{split}_hidden.npy", mode="w+", dtype=np.float16,
                                           shape=(len(records), *first.shape[1:]))
        row = 0
        for stem in shards:
            part = np.load(stem.with_suffix(".npy"), mmap_mode="r")
            merged[row:row + len(part)] = part
            row += len(part)
        merged.flush()
        (out_dir / "layers.json").write_text(json.dumps(self.layers))
        path = write_records(out_dir / f"{split}_records.json", records)
        n_wrong = sum(r.wrong for r in records)
        print(f"{split}: {len(records)} answers, {n_wrong} wrong ({n_wrong / len(records):.1%}); wrote {path}")
        return path


def run(splits: Sequence[str] = ("train", "val", "test"), limit: int | None = None, batch_size: int = 8,
        model_dir: Path | None = None, device: str = "auto") -> dict[str, Path]:
    """Load Qwen once and answer every split's questions into outputs/answers/; limit applies to each split."""
    vlm = VLM(model_dir, device=device)
    return {split: vlm.run(paths.queries_file(split), split, limit=limit, batch_size=batch_size) for split in splits}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--download", action="store_true", help=f"download {MODEL_ID} into src/model/models/")
    parser.add_argument("--run", action="store_true", help="answer the question files (stage 4)")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="with --run")
    parser.add_argument("--limit", type=int, help="with --run: first N questions of each split (a pilot)")
    parser.add_argument("--batch-size", type=int, default=8, help="with --run")
    parser.add_argument("--model-dir", type=Path, help="weights folder (default: src/model/models/Qwen2.5-VL-3B-Instruct)")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--question")
    parser.add_argument("--subj-box", type=int, nargs=4, metavar=("X", "Y", "W", "H"), help="drawn in red")
    parser.add_argument("--obj-box", type=int, nargs=4, metavar=("X", "Y", "W", "H"), help="drawn in blue")
    parser.add_argument("--device", default="auto", help="auto (cuda, then mps, then cpu), or cuda, mps or cpu")
    args = parser.parse_args()

    if args.download:
        print(f"downloaded to {download(args.model_dir)}")
        return
    if args.run:
        run(args.splits, args.limit, args.batch_size, args.model_dir, args.device)
        return
    if not (args.image and args.question):
        parser.error("give --download, --run, or --image and --question")

    image = Image.open(args.image)
    if args.subj_box and args.obj_box:
        image = draw_boxes(image, BBox(**dict(zip("xywh", args.subj_box))), BBox(**dict(zip("xywh", args.obj_box))))
    vlm = VLM(args.model_dir, device=args.device)
    answer = vlm.ask([image], [args.question])[0]
    print(f"answer {answer.answer}  p_yes {answer.p_yes:.4f}  p_no {answer.p_no:.4f}  "
          f"hidden {answer.hidden.shape} from layers {vlm.layers}")


if __name__ == "__main__":
    main()
