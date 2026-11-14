# VSR review worksheet

## Purpose

Isaac will visually review the 941 directional VSR candidate statements before
larger feature extraction. The assistant handles data preparation and downloads.
This is a local research worksheet, with a clean presentation requested by the user.

## Workflow and constraints

- Show original images, source truth labels and proposed transformations.
- Offer approve-all-checks, original-only and set-aside decisions.
- Preserve source labels and image-disjoint splits.
- Save human decisions locally; expose no VLM predictions or agent approvals.
- Require all statements to be reviewed before greenlighting extraction.
- Greenlighting prepares manifests; it does not launch a GPU job.

## Implementation scope

A small local Python server and plain HTML/CSS/JavaScript fit the existing Python
project. No external hosting or frontend installation is needed. This is a narrow
utility surface, not a new public product or branding exercise.
