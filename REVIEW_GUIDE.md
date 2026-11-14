# Review the larger VSR dataset

Open **http://127.0.0.1:8766/** in your browser. All 855 images are already local
and verified. You are reviewing **941 statements**, organised into 19 batches of
up to 50. Multiple statements can use the same image; review each statement.

## Start

1. Open **Review guide — read this before starting** on the page.
2. At **Finish the review**, enter your name and select **Save name** once.
3. Choose **Batch 1 · 1–50** at the top, then work through the statements.

No VLM answers or earlier agent flags are shown. All human decisions start as
unreviewed, including the 100 pilot statements. Your decisions save into the
project folder each time you use a decision button. You can stop and resume;
use **Next unreviewed** to continue. **Export backup** downloads a copy of your
saved decisions.

## For each statement

1. **Find both objects.** They should be identifiable enough to judge the relation.
   Use **Open full size** if needed. The page preserves the complete image.
2. **Check the VSR source label.** Does True or False make sense for the original
   statement? Keep that source label unchanged. A questionable label gets set aside.
3. **Read the converse.** Objects and relation swap: “The cat is above the table”
   becomes “The table is below the cat.” Under the same interpretation, their truth
   should agree. Check the same two object instances are intended.
4. **Check the mirrored image and statement.** A horizontal reflection swaps
   left/right, while above/below stay unchanged. The transformed statement should
   retain the original truth. Perspective-sensitive cases need care.
5. **Read the exclusive statement.** Its relation should be incompatible with the
   original relation for those same objects. You do not assign a truth label to it.
   A false original and its opposite can BOTH be false.

For left/right, judge whether a consistent viewer-relative interpretation works.
If the source label only makes sense from an object's own perspective, choose
Original only when the original itself is otherwise usable. If you cannot decide
whether the source label is reasonable, choose Set aside instead.

## Choose a decision

| Decision | When to choose it | What happens later |
|---|---|---|
| **Approve all checks** / **A** | Objects, source label and transformations are suitable | Original + grey-image + three consistency scores |
| **Original only** / **O** | Original question/label is usable; a transformation is uncertain | Original + grey-image scores; no consistency features |
| **Set aside** / **S** | Missing/unclear objects, questionable label or an unresolved spatial interpretation | Excluded from the reviewed extraction dataset; preserved in the source data |

For Original only or Set aside, tick at least one reason first. If you select
Other, add a brief note. Do not retain issue checkboxes when approving all checks.
If objects or the source label are questionable, use Set aside rather than
Original only. Do not set aside a valid False statement merely because it is False.

Decisions save before advancing. The left/right arrow keys move between cases.
Keyboard shortcuts are ignored while typing in a field. **Clear decision** returns
a case to unreviewed. Notes save with a decision or with **Save note**.

Use the same criteria throughout the dataset. Do not choose cases based on whether
you think the VLM will answer them correctly. The train/validation/test split is
already fixed and is preserved automatically.

## Greenlight the dataset

1. Every statement must have a decision: the counter must read **941 of 941**.
   Set-aside cases count as reviewed and remain excluded; there is no need to force
   uncertain cases into the approved group.
2. At **Finish the review**, check your saved reviewer name.
3. Tick **I applied the same criteria throughout. Set-aside cases will stay excluded.**
4. Select **Greenlight feature extraction**.
5. Confirm the page says **Dataset greenlit**, then tell Codex that your review is
   complete so feature extraction can begin.

Greenlighting validates the original metadata and image checksums, then writes
approved, original-only and set-aside manifests, including separate train,
validation and test files. It does not run the VLM. Original VSR labels never change;
the VLM error labels will be created automatically after inference.

Any later saved edit invalidates the greenlight. Review your changes and greenlight
again before extraction.

## Restart the review page if needed

In PowerShell:

```powershell
cd C:\Users\Isaac\4001Project
.\.venv\Scripts\python.exe scripts\serve_vsr_review.py
```

Keep that PowerShell window open, then visit **http://127.0.0.1:8766/**. Closing the
browser does not erase saved decisions. If the server is already running, open the
URL directly rather than starting a second copy.

## Files

- Candidate dataset: `data/vsr/processed/directional_candidates.jsonl`.
- Images: `data/vsr/images/`.
- Review dataset and checksums: `data/vsr/review/dataset.json`.
- Your saved decisions: `data/vsr/review/human_decisions.json` (created on first save).
- Approved manifests: `data/vsr/review/approved/revision_XXXXX/` after greenlighting.

The current greenlight recorded in `human_decisions.json` identifies the active
approved manifests. Earlier approval folders are retained as history; they must
not be mistaken for the active approval after later edits.
