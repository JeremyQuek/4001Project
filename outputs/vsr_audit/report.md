# VSR pilot: first-pass visual audit

Reviewed 100 statements across 98 training images. 73 have no immediate issue; 27 are held for closer review.

Reviewer: Codex visual inspection. This is one agent-assisted review, not independent human annotation or inter-rater validation. All original VSR labels are preserved. A flag indicates uncertainty, not proof that an annotation is wrong. No VLM predictions were used to select or flag examples.

## Transformation rules

- Converse: swap subject/object and swap left/right or above/below. Horizontal equivalence requires a consistent viewer-relative interpretation.
- Exclusive: replace the relation with its opposite, but do not assign a new truth label. False original statements can have false opposites.
- Mirror: horizontally reflect pixels, swap left/right in the statement, and leave above/below unchanged. Perspective-sensitive cases remain held.
- Blind: use the original statement with an equally sized grey image; this diagnostic has no independent ground-truth answer.

## Files and use

`reviewed_pilot.jsonl` retains all 100 examples and review flags. `consistency_ready_pilot.jsonl` is a provisional subset for initial consistency feature checks. `flagged_pilot.jsonl` records held cases. `review.csv` opens as a review table in Excel. To revise decisions reproducibly, edit `configs/vsr_pilot_audit.json` and rerun `scripts/finalize_vsr_audit.py`.

Run original-answer evaluation on the full pilot with VSR labels. For consistency ablations, compare all feature groups on the same reviewed subset; do not compare a baseline on 100 examples with a verifier on the smaller subset. Report both counts and the selection criteria. This review does not relabel or delete the source dataset.

The numeric reference-frame codes are preserved without assuming their meanings. The two highlighted perspective cases are flagged using image evidence and uncertain converse semantics.

## Cases requiring closer review

| Audit # | Statement | VSR truth | Issue |
|---|---|---:|---|
| [015](C:/Users/Isaac/4001Project/data/vsr/images/000000316089.jpg) | The bench is above the cat. | 0 | The cat is on the bench seat, while the backrest extends above it. The reference part of the bench is ambiguous. |
| [017](C:/Users/Isaac/4001Project/data/vsr/images/000000034865.jpg) | The dining table is above the dog. | 1 | Only part of a white dog is visible at the far right edge beside the person; full object grounding is difficult. |
| [022](C:/Users/Isaac/4001Project/data/vsr/images/000000527906.jpg) | The book is below the sandwich. | 0 | The apparent book is partly hidden behind the cup in the background; verify the intended book referent. |
| [025](C:/Users/Isaac/4001Project/data/vsr/images/000000099129.jpg) | The motorcycle is right of the bus. | 1 | The cycles on the right of the bus appear to be bicycles; a motorcycle is visible at the left edge. Verify the intended motorcycle. |
| [027](C:/Users/Isaac/4001Project/data/vsr/images/000000554102.jpg) | The cup is left of the person. | 1 | The cup is on the viewer-right of the person, although VSR labels left-of as true. A person-intrinsic reading is plausible; swapped-object equivalence is unsafe. |
| [033](C:/Users/Isaac/4001Project/data/vsr/images/000000473720.jpg) | The pizza is left of the dog. | 1 | Pizza is in front of/below the dog, with substantial horizontal overlap. Source reference_frame is 0.0; viewer-relative converse assumptions need review. |
| [034](C:/Users/Isaac/4001Project/data/vsr/images/000000319257.jpg) | The cat is left of the potted plant. | 0 | The cat and foreground cactus branches overlap; the potted plant is partly outside the image and its horizontal referent is unclear. |
| [035](C:/Users/Isaac/4001Project/data/vsr/images/000000095100.jpg) | The pizza is left of the refrigerator. | 0 | The foreground pizza appears partly viewer-left of the refrigerator despite the False label. Depth and horizontal overlap make interpretation uncertain. |
| [036](C:/Users/Isaac/4001Project/data/vsr/images/000000170442.jpg) | The suitcase is above the bed. | 0 | The suitcase is on a chair beside the bed. Image height and physical height yield different intuitions about above. |
| [039](C:/Users/Isaac/4001Project/data/vsr/images/000000045283.jpg) | The hot dog is below the person. | 0 | Only a hand/arm of the person is visible; the vertical position of the whole person cannot be established reliably. |
| [040](C:/Users/Isaac/4001Project/data/vsr/images/000000550019.jpg) | The car is below the parking meter. | 0 | The car is behind the parking meter; their vertical extents overlap. Below depends on the chosen object reference point. |
| [041](C:/Users/Isaac/4001Project/data/vsr/images/000000162427.jpg) | The laptop is right of the cat. | 1 | Only the cat paw is visible on the laptop; the whole cat position is not established. |
| [049](C:/Users/Isaac/4001Project/data/vsr/images/000000284589.jpg) | The bed is below the person. | 0 | A person is not clearly identifiable in the bedroom image at the inspected resolution. Verify the intended referent. |
| [055](C:/Users/Isaac/4001Project/data/vsr/images/000000329502.jpg) | The couch is above the snowboard. | 1 | The presumed snowboard is mostly cropped at the lower edge. Confirm that this object is a snowboard. |
| [059](C:/Users/Isaac/4001Project/data/vsr/images/000000526534.jpg) | The zebra is left of the elephant. | 0 | The zebra and elephant are partially occluded/cropped in a display; horizontal object reference points are unclear. |
| [068](C:/Users/Isaac/4001Project/data/vsr/images/000000329502.jpg) | The couch is below the snowboard. | 0 | Same image as audit 55; presumed snowboard is mostly cropped at the lower edge and needs object identification. |
| [075](C:/Users/Isaac/4001Project/data/vsr/images/000000035211.jpg) | The dog is left of the bicycle. | 1 | Only a small bicycle wheel segment is visible at the right edge; the full bicycle referent requires confirmation. |
| [078](C:/Users/Isaac/4001Project/data/vsr/images/000000377772.jpg) | The potted plant is above the bed. | 0 | Two bunk-bed levels are visible. Above can differ depending on which bed is intended. |
| [079](C:/Users/Isaac/4001Project/data/vsr/images/000000535185.jpg) | The tv is above the laptop. | 0 | The laptop dominates the foreground; the alleged TV is blurred in the background and cannot be reliably located. |
| [082](C:/Users/Isaac/4001Project/data/vsr/images/000000065806.jpg) | The person is right of the dining table. | 0 | Only hands/forearms of the person are visible beside the table; whole-person horizontal location is uncertain. |
| [086](C:/Users/Isaac/4001Project/data/vsr/images/000000084959.jpg) | The banana is left of the person. | 1 | I cannot confidently identify the banana in the kitchen image; the person obscures part of the left background. |
| [087](C:/Users/Isaac/4001Project/data/vsr/images/000000550536.jpg) | The broccoli is below the person. | 0 | Broccoli is in the foreground below the visible torso, despite the False label. The person is cropped and physical versus projected height is uncertain. |
| [090](C:/Users/Isaac/4001Project/data/vsr/images/000000204351.jpg) | The apple is above the banana. | 0 | The visible food looks like sandwiches and fries; I cannot identify an apple or banana. The False label may be valid as an absent-object negative, but this is not a clean spatial example. |
| [092](C:/Users/Isaac/4001Project/data/vsr/images/000000137281.jpg) | The stop sign is above the bicycle. | 0 | The prominent sign is a speed display; a small distant red sign may be the stop sign. Verify object identity and position. |
| [094](C:/Users/Isaac/4001Project/data/vsr/images/000000084324.jpg) | The handbag is right of the bed. | 0 | The handbag fills most of the image; the bed is not clearly identifiable, making right-of difficult to validate. |
| [096](C:/Users/Isaac/4001Project/data/vsr/images/000000532667.jpg) | The bird is below the umbrella. | 1 | A dark bird-like shape overlaps the umbrella fabric and branches. Both bird identity and a clean below relation need confirmation. |
| [097](C:/Users/Isaac/4001Project/data/vsr/images/000000389584.jpg) | The teddy bear is below the dog. | 0 | The dog and teddy bear overlap closely; head position and whole-body position give different vertical readings. |

## Contact sheets

- [Examples 001–010](C:/Users/Isaac/4001Project/outputs/vsr_audit/contact_01.jpg)
- [Examples 011–020](C:/Users/Isaac/4001Project/outputs/vsr_audit/contact_02.jpg)
- [Examples 021–030](C:/Users/Isaac/4001Project/outputs/vsr_audit/contact_03.jpg)
- [Examples 031–040](C:/Users/Isaac/4001Project/outputs/vsr_audit/contact_04.jpg)
- [Examples 041–050](C:/Users/Isaac/4001Project/outputs/vsr_audit/contact_05.jpg)
- [Examples 051–060](C:/Users/Isaac/4001Project/outputs/vsr_audit/contact_06.jpg)
- [Examples 061–070](C:/Users/Isaac/4001Project/outputs/vsr_audit/contact_07.jpg)
- [Examples 071–080](C:/Users/Isaac/4001Project/outputs/vsr_audit/contact_08.jpg)
- [Examples 081–090](C:/Users/Isaac/4001Project/outputs/vsr_audit/contact_09.jpg)
- [Examples 091–100](C:/Users/Isaac/4001Project/outputs/vsr_audit/contact_10.jpg)

[Original/mirror comparison examples](C:/Users/Isaac/4001Project/outputs/vsr_audit/mirror_examples.jpg)

## Next step

Run a GPU pilot that preserves all original answers/labels and records five diagnostic scores where the transformations are eligible. Keep held cases separate when interpreting logical inconsistencies. Choose final processing settings using training pilot data before extracting validation/test features.
