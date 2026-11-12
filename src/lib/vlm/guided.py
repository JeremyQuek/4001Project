"""Token-by-token greedy generation with an optional hallucination check on each object word.

Policies
  none       plain greedy decoding through the same loop (the "no correction" reference, so
             every Part 3 condition shares one decoding path).
  candidate  (design A) once an object word is complete, score the hidden state at its first
             token T (the `post` state, which has seen the candidate word). If flagged, roll the
             KV cache back to T, ban that token at slot T, and re-pick. The new candidate is
             checked the same way.
  skip       (design B) score the `pre1` state (the state that chose token T). If flagged, roll
             back to T and force a sentence end there, so no object is written in that slot.

A mention is only judged once the word after it is complete, so multi-word objects
("teddy bear", "passenger train") are resolved before the check.
"""
import random
import time

import torch

from .model import encode, image_messages, reset_rope
from .tokens import char_to_token


class Flagger:
    """Decides whether a completed object mention is a hallucination.

    source: "probe" | "random" | "perfect"; for random, `rate` is the probe's validation flag rate."""

    def __init__(self, source, probe=None, gt=None, rate=0.0, seed=0):
        self.source, self.probe, self.gt, self.rate = source, probe, gt, rate
        self.rng = random.Random(seed)

    def __call__(self, mention, state):
        if self.source == "probe":
            return float(self.probe.score(state[None])[0]) >= self.probe.threshold
        if self.source == "perfect":
            return mention.category not in self.gt
        if self.source == "random":
            return self.rng.random() < self.rate
        raise ValueError(self.source)


def _final_mentions(chair, text):
    """Mentions followed by at least one complete token (or punctuation), i.e. no longer extendable."""
    toks = chair.tokenize(text)
    out = []
    for m in chair.find_mentions(text):
        after = [t for t in toks if t[1] >= m.end]
        if any((not t[0].isalnum()) or t[2] < len(text) for t in after):
            out.append(m)
    return out


@torch.no_grad()
def guided_generate(model, processor, chair, image_path, prompt, cfg, max_new_tokens,
                    policy="none", flagger=None, probe_layer=None,
                    max_repicks_per_slot=3, max_repicks_per_caption=10, stops=()):
    from transformers import DynamicCache

    tok = processor.tokenizer
    need_hidden = policy in ("candidate", "skip") and flagger is not None and flagger.source == "probe"
    period_id = tok.encode(".", add_special_tokens=False)[0]

    t0 = time.time()
    batch = encode(processor, [image_messages(image_path, prompt, cfg)], model.device)
    P = batch["input_ids"].shape[1]
    cache = DynamicCache()
    reset_rope(model)
    out = model(**batch, past_key_values=cache, use_cache=True,
                cache_position=torch.arange(P, device=model.device),
                output_hidden_states=need_hidden)

    ids, ends = [], []
    # logits_at[j]: next-token logits for caption slot j; states[j]: probe-layer state at position P-1+j.
    logits_at = [out.logits[0, -1].float()]
    states = [out.hidden_states[probe_layer][0, -1].float().cpu().numpy()] if need_hidden else []
    bans = {}        # slot -> set of banned token ids
    forced = {}      # slot -> forced token id (skip policy)
    accepted = set() # first-token slots already judged and kept
    repicks = {}
    n_repicks = 0
    interventions = []

    def rollback(T):
        nonlocal ids, ends, logits_at, states
        cache.crop(P + T)
        ids, ends = ids[:T], ends[:T]
        logits_at = logits_at[:T + 1]
        if need_hidden:
            states = states[:T + 1]
        for s in [s for s in accepted if s >= T]:
            accepted.discard(s)

    finished = False
    while not finished:
        j = len(ids)
        if j >= max_new_tokens:
            finished = True
        else:
            if j in forced:
                nxt = forced.pop(j)
            else:
                logits = logits_at[j].clone()
                for b in bans.get(j, ()):
                    logits[b] = float("-inf")
                nxt = int(torch.argmax(logits))
            if nxt in stops:
                finished = True
            else:
                ids.append(nxt)
                ends.append(len(tok.decode(ids, skip_special_tokens=True)))
                step = model(input_ids=torch.tensor([[nxt]], device=model.device),
                             attention_mask=torch.ones(1, P + j + 1, device=model.device, dtype=torch.long),
                             past_key_values=cache, use_cache=True,
                             cache_position=torch.tensor([P + j], device=model.device),
                             output_hidden_states=need_hidden)
                logits_at.append(step.logits[0, -1].float())
                if need_hidden:
                    states.append(step.hidden_states[probe_layer][0, -1].float().cpu().numpy())

        if policy == "none" or flagger is None:
            continue

        # Judge newly completed object mentions (all remaining ones once generation has stopped).
        text = tok.decode(ids, skip_special_tokens=True)
        mentions = chair.find_mentions(text) if finished else _final_mentions(chair, text)
        for m in mentions:
            T = char_to_token(ends, m.start)
            if T in accepted:
                continue
            state = None
            if need_hidden:
                state = states[T + 1] if policy == "candidate" else states[T]
            flagged = flagger(m, state)
            budget_ok = repicks.get(T, 0) < max_repicks_per_slot and n_repicks < max_repicks_per_caption
            if flagged and budget_ok:
                interventions.append({"slot": T, "word": m.word, "category": m.category, "policy": policy})
                repicks[T] = repicks.get(T, 0) + 1
                n_repicks += 1
                old = ids[T]
                rollback(T)
                if policy == "candidate":
                    bans.setdefault(T, set()).add(old)
                else:
                    forced[T] = period_id
                finished = False
                break
            accepted.add(T)

    return {
        "ids": ids,
        "caption": tok.decode(ids, skip_special_tokens=True),
        "n_interventions": n_repicks,
        "interventions": interventions,
        "seconds": time.time() - t0,
    }
