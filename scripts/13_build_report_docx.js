// Stage 13: build the written report (Word, A4, Arial 10 pt) from saved artifacts.
//
// Every number in a table or sentence is read from outputs/*.json, the dataset
// metadata, or the teammate's committed test_metrics.csv (branch origin/isaac).
// Run from the repository root after stages 8, 10 and 12:
//   node scripts/13_build_report_docx.js        (needs the `docx` npm package)
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const {
  AlignmentType, BorderStyle, Document, HeadingLevel, ImageRun, LevelFormat, Packer, PageBreak,
  Paragraph, ShadingType, Table, TableCell, TableRow, TextRun, WidthType, Footer, PageNumber,
} = require('docx');

const ROOT = path.resolve(__dirname, '..');
const OUT = path.join(ROOT, 'outputs');
const read = (file) => JSON.parse(fs.readFileSync(path.join(ROOT, file), 'utf8'));
const results = read('outputs/results.json');
const extra = read('outputs/aggregation.json');
const frozen = read('outputs/detectors/frozen.json');
const meta = read('data/vg/detector_ready/metadata.json');
const filters = read('data/vg/processed/filter_summary.json');
const extraction = read('outputs/extraction_summary.json');

// Teammate's VSR test metrics, straight from their committed file.
const vsr = {};
execSync('git show origin/isaac:outputs/mlp_comparison/test_metrics.csv', { cwd: ROOT }).toString().trim()
  .split('\n').slice(1).forEach((line) => {
    const c = line.split(',');
    vsr[`${c[0]}/${c[1]}`] = { rows: +c[2], errors: +c[3], auroc: +c[4], low: +c[7], high: +c[8],
      coverage: +c[11], kept: +c[13] };
  });

// ---- number formatting ----
const f3 = (v) => v.toFixed(3);
const pc = (v) => `${(100 * v).toFixed(1)}%`;
const sg = (v) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(3)}`;
const ci = (p) => `[${f3(p[0])}, ${f3(p[1])}]`;
const sci = (p) => `[${sg(p[0])}, ${sg(p[1])}]`;
const pts = (v) => `${v >= 0 ? '+' : '−'}${Math.abs(100 * v).toFixed(1)}`;
const int = (v) => v.toLocaleString('en-US');

// ---- shortcuts into the artifacts ----
const test = results.cohorts.all;
const M = test.models;
const CMP = test.bootstrap.comparisons;
const methods = extra.methods;
const routers = extra.cascade.routers;
const rules = extra.aggregation_rules.rules;
const DET = extra.detector_used;                      // e.g. mlp_all_hidden36
const LAYER = DET.split('hidden')[1];
const single = methods.single_no_abstention.points[0].accuracy_on_answered;
const agg = methods.aggregate.points[0].accuracy_on_answered;
const gain = extra.aggregate_minus_single;
const at = (name, target) => methods[name].points.find((p) => p.target_coverage === target);
const route = (name, share) => routers[name].points.find((p) => Math.abs(p.target_escalation - share) < 1e-9);
const half = route(DET, 0.5);
const most = route(DET, 0.7);
const op = (name) => M[name].operating_point;
const byPred = results.breakdowns.canonical_predicate;
const byVar = results.breakdowns.variant;
const mlpVsConf = CMP[`${DET}_minus_mlp_all_confidence`];
const lrVsConf = CMP[`lr_all_hidden${LAYER}_minus_lr_all_confidence`];
const mlpVsLr = CMP[`all_hidden${LAYER}_mlp_minus_lr`];
const mlpVsHeur = CMP[`all_hidden${LAYER}_mlp_minus_heuristic`];

// ---- document helpers ----
const FONT = 'Arial';
const WIDTH = 9638;                                   // A4 with 2 cm margins, in DXA
function runs(text, base = {}) {
  // **bold**, *italic*, [[placeholder to fill in]] (highlighted)
  return text.split(/(\*\*[^*]+\*\*|\*[^*]+\*|\[\[[^\]]+\]\])/).filter(Boolean).map((part) => {
    if (part.startsWith('**')) return new TextRun({ ...base, text: part.slice(2, -2), bold: true });
    if (part.startsWith('[[')) return new TextRun({ ...base, text: part.slice(2, -2), highlight: 'yellow' });
    if (part.startsWith('*')) return new TextRun({ ...base, text: part.slice(1, -1), italics: true });
    return new TextRun({ ...base, text: part });
  });
}
const P = (text, opts = {}) => new Paragraph({ children: runs(text), spacing: { after: 100 },
  alignment: AlignmentType.JUSTIFIED, ...opts });
const H1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
const H2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
const B = (text) => new Paragraph({ children: runs(text), numbering: { reference: 'bullets', level: 0 },
  spacing: { after: 40 } });
const caption = (text) => new Paragraph({ children: runs(text, { italics: true }), spacing: { before: 60, after: 140 } });
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

function table(header, rows, weights) {
  const total = weights.reduce((a, b) => a + b, 0);
  const widths = weights.map((w) => Math.floor((WIDTH * w) / total));
  widths[widths.length - 1] += WIDTH - widths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: 'A6A6A6' };
  const cell = (text, i, head) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    borders: { top: border, bottom: border, left: border, right: border },
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    shading: head ? { type: ShadingType.CLEAR, fill: 'E8EEF7', color: 'auto' } : undefined,
    children: [new Paragraph({ children: runs(String(text), head ? { bold: true } : {}),
      alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })],
  });
  return new Table({
    width: { size: WIDTH, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, true)) }),
      ...rows.map((row) => new TableRow({ cantSplit: true, children: row.map((c, i) => cell(c, i, false)) }))],
  });
}
function figure(file, width, height) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80 }, keepNext: true,
    children: [new ImageRun({ type: 'png', data: fs.readFileSync(path.join(OUT, file)),
      transformation: { width, height } })] });
}

// ---- tables built from artifacts ----
const INPUTS = [['Confidence [p_yes, confidence]', 'confidence'], ['Hidden layer 9', 'hidden9'],
  ['Hidden layer 18', 'hidden18'], ['Hidden layer 36', 'hidden36']];
const aurocCell = (name) => `${f3(M[name].auroc)} ${ci(M[name].auroc_image_bootstrap95)}`;
const detectionRows = [['1 − confidence (nothing fitted)', aurocCell('heuristic_confidence'),
  f3(M.heuristic_confidence.average_precision), '—', '—']]
  .concat(INPUTS.map(([label, key]) => [label, aurocCell(`lr_all_${key}`), f3(M[`lr_all_${key}`].average_precision),
    aurocCell(`mlp_all_${key}`), f3(M[`mlp_all_${key}`].average_precision)]));

const abstainRow = (label, name) => [label, pc(op(name).coverage), pc(op(name).retained_vlm_accuracy),
  `${op(name).errors_caught} / ${test.errors}`, pc(op(name).abstention_precision)];
const abstainRows = [['No abstention', '100.0%', pc(test.unfiltered_vlm_accuracy), `0 / ${test.errors}`, '—'],
  abstainRow('1 − confidence', 'heuristic_confidence'), abstainRow('LR, confidence', 'lr_all_confidence'),
  abstainRow(`LR, hidden layer ${LAYER}`, `lr_all_hidden${LAYER}`), abstainRow(`MLP, hidden layer ${LAYER}`, DET)];

const cov = (name, t) => `${pc(at(name, t).accuracy_on_answered)} @ ${pc(at(name, t).coverage)}`;
const usageRows = [
  ['Single query (baseline)', '1', pc(single), '—', '—'],
  ['Abstain on 1 − confidence', '1', pc(single), cov('abstain_confidence', 0.8), cov('abstain_confidence', 0.6)],
  [`Abstain on layer-${LAYER} MLP risk`, '1', pc(single), cov('abstain_detector', 0.8), cov('abstain_detector', 0.6)],
  ['Aggregate four variants', '4', pc(agg), '—', '—'],
  ['Aggregate, abstain on small margin', '4', pc(agg), cov('aggregate_abstain_margin', 0.8),
    cov('aggregate_abstain_margin', 0.6)],
  [`Aggregate, abstain on mean MLP risk`, '4', pc(agg), cov('aggregate_abstain_detector', 0.8),
    cov('aggregate_abstain_detector', 0.6)]];

const cascadeRows = [0, 0.2, 0.4, 0.5, 0.7, 1.0].map((share) => {
  const m = route(DET, share);
  return [m.mean_vlm_calls.toFixed(1), pc(share), pc(m.random_router_accuracy), pc(route('confidence', share).accuracy),
    pc(route(`lr_all_hidden${LAYER}`, share).accuracy), pc(m.accuracy),
    m.with_abstention ? pc(m.with_abstention.accuracy_on_answered) : '—'];
});

const ruleRows = Object.values(rules).map((r) => {
  const c = r.cascade.find((p) => Math.abs(p.target_escalation - 0.5) < 1e-9);
  const base = r.label.startsWith('Confidence only');
  return [r.label, pc(r.accuracy), base ? 'baseline'
    : `${pts(r.minus_confidence_rule)} [${pts(r.minus_confidence_rule_image_bootstrap95[0])}, ${pts(r.minus_confidence_rule_image_bootstrap95[1])}]`,
    pc(r.false_yes_rate), pc(r.abstention[0].accuracy_on_answered), pc(c.accuracy)];
});

const v = (key) => vsr[`all/${key}`];
const vsrRows = [
  ['Test questions / VLM errors', `${v('heuristic_confidence').rows} / ${v('heuristic_confidence').errors}`, `${test.rows} / ${test.errors}`],
  ['VLM accuracy without abstention', pc(1 - v('heuristic_confidence').errors / v('heuristic_confidence').rows), pc(test.unfiltered_vlm_accuracy)],
  ['AUROC, 1 − confidence', `${f3(v('heuristic_confidence').auroc)} [${f3(v('heuristic_confidence').low)}, ${f3(v('heuristic_confidence').high)}]`, aurocCell('heuristic_confidence')],
  ['AUROC, LR on confidence', `${f3(v('lr_all_confidence').auroc)} [${f3(v('lr_all_confidence').low)}, ${f3(v('lr_all_confidence').high)}]`, aurocCell('lr_all_confidence')],
  ['AUROC, MLP on hidden layer 9', f3(v('mlp_all_hidden9').auroc), f3(M.mlp_all_hidden9.auroc)],
  ['AUROC, MLP on hidden layer 18', f3(v('mlp_all_hidden18').auroc), f3(M.mlp_all_hidden18.auroc)],
  ['AUROC, MLP on hidden layer 36', `${f3(v('mlp_all_hidden36').auroc)} [${f3(v('mlp_all_hidden36').low)}, ${f3(v('mlp_all_hidden36').high)}]`, aurocCell('mlp_all_hidden36')],
  ['Abstention on 1 − confidence', `${pc(v('heuristic_confidence').kept)} @ ${pc(v('heuristic_confidence').coverage)}`, `${pc(op('heuristic_confidence').retained_vlm_accuracy)} @ ${pc(op('heuristic_confidence').coverage)}`],
  ['Abstention on layer-36 MLP risk', `${pc(v('mlp_all_hidden36').kept)} @ ${pc(v('mlp_all_hidden36').coverage)}`, `${pc(op('mlp_all_hidden36').retained_vlm_accuracy)} @ ${pc(op('mlp_all_hidden36').coverage)}`],
];

// Appendix tables
const comparisonRows = Object.values(CMP).map((c) => [`${c.model_a} − ${c.model_b}`, sg(c.auroc_delta),
  sci(c.paired_image_bootstrap95), (c.paired_image_bootstrap95[0] > 0 || c.paired_image_bootstrap95[1] < 0) ? 'excludes 0' : 'includes 0']);
const selectedRows = Object.entries(frozen.models).filter(([, e]) => e.kind !== 'heuristic').map(([name, e]) => (e.kind === 'lr'
  ? [name, `C = ${e.selected_C}`, e.selected_class_weight || 'none', f3(e.validation_metrics.auroc), f3(M[name].auroc), '—']
  : [name, `decay = ${e.selected_weight_decay}`, e.selected_class_weight || 'none',
    f3(e.seeds[String(frozen.mlp_config.primary_seed)].validation_metrics.auroc), f3(M[name].auroc),
    `${f3(test.seed_stability[name].test_auroc_mean)} ± ${f3(test.seed_stability[name].test_auroc_std)}`]));
const PREDICATES = Object.keys(extra.by_predicate);
const predicateRows = PREDICATES.map((p) => [p, pc(byPred[p].vlm_accuracy), pc(byPred[p].vlm_yes_rate),
  f3(byPred[p].auroc.heuristic_confidence), f3(byPred[p].auroc[`lr_all_hidden${LAYER}`]), f3(byPred[p].auroc[DET]),
  pc(extra.by_predicate[p].single_accuracy), pc(extra.by_predicate[p].aggregate_accuracy)]);
const variantRows = ['orig', 'converse', 'swap', 'opposite'].map((k) => [k, pc(byVar[k].vlm_accuracy), pc(byVar[k].vlm_yes_rate),
  f3(byVar[k].auroc.heuristic_confidence), f3(byVar[k].auroc.lr_all_confidence), f3(byVar[k].auroc[`lr_all_hidden${LAYER}`]),
  f3(byVar[k].auroc[DET])]);

const worstPred = PREDICATES.reduce((a, b) => (byPred[a].vlm_accuracy < byPred[b].vlm_accuracy ? a : b));
const bestPred = PREDICATES.reduce((a, b) => (byPred[a].vlm_accuracy > byPred[b].vlm_accuracy ? a : b));
const cons = extra.consistency;
const err = extra.error_types;
const tw = rules.trust_weighted;
const dv = rules.detector_votes;

// ---- the report ----
const cover = [
  new Paragraph({ spacing: { before: 2400, after: 200 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'SC4001 Neural Networks and Deep Learning', size: 28 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
    children: [new TextRun({ text: 'Group Project E: Hallucination and Reliability in Vision Language Models', size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
    children: [new TextRun({ text: 'Detecting and Reducing Spatial-Relation Errors of a Frozen Vision Language Model', size: 36, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 1200 },
    children: [new TextRun({ text: 'Hidden-state error detectors, abstention, self-consistency aggregation and a detect-then-verify cascade', size: 24, italics: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: runs('[[Group number: ___]]') }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: runs('[[Member 1 full name (matriculation number)]]') }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: runs('[[Member 2 full name (matriculation number)]]') }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 }, children: runs('[[Member 3 full name (matriculation number)]]') }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: runs('Preliminary draft, 15 November 2026') }),
  pageBreak(),
];

const SECTIONS = ['1. Introduction', '2. Review of existing techniques', '3. Methods', '4. Experiments and results',
  '5. Discussion', '6. AI Use Statement', 'References', 'Appendix A. Reproducing the results',
  'Appendix B. Selected hyperparameters', 'Appendix C. Paired AUROC comparisons', 'Appendix D. Breakdowns by predicate and variant'];
const contents = [
  new Paragraph({ spacing: { after: 160 }, children: [new TextRun({ text: 'Contents', bold: true, size: 28 })] }),
  ...SECTIONS.map((s) => new Paragraph({ children: runs(s), spacing: { after: 60 } })),
  P('[[Replace this list with an automatic table of contents (Insert > Table of contents) before exporting to PDF, so that page numbers appear.]]', { spacing: { before: 200 } }),
  pageBreak(),
];

const body = [
  H1('1. Introduction'),
  P('Vision language models (VLMs) answer questions about images fluently, but they also assert things that the image does not show. Most work on such hallucinations concerns objects: the model names an object that is absent [4, 5]. Relations between objects are a harder and less studied case [6, 7]. A model may correctly see a cup and a table and still say that the table is on the cup. Spatial relations matter in practice, for example in robotics, navigation and assistive tools, where a confident wrong answer is worse than no answer.'),
  P('This project studies one frozen VLM, Qwen2.5-VL-3B-Instruct [3], on Yes/No questions about spatial relations, and asks three questions.'),
  B('**RQ1 (detection).** Can a small neural network, reading the VLM’s confidence or its hidden state, predict when the VLM’s answer is wrong?'),
  B('**RQ2 (where the signal is).** Is that information already in the output confidence, and at which depth of the network does it appear?'),
  B('**RQ3 (use).** Once errors can be flagged, how should the flag be used: to withhold answers (abstention), to cross-check the model against itself (aggregation), or to decide when the cross-check is worth its cost (a cascade)?'),
  P(`We treat a hallucination operationally as a wrong Yes/No answer, and in the strict sense as a “Yes” to a false statement. Our main contributions are: (i) a reproducible pipeline that turns Visual Genome [1] relationship annotations into ${int(meta.counts.train.rows + meta.counts.val.rows + meta.counts.test.rows)} balanced spatial questions; (ii) a controlled comparison of logistic-regression and MLP error detectors on confidence and on three hidden layers, with a single held-out test evaluation and image-clustered confidence intervals; (iii) evidence that the final hidden layer carries error information beyond the output confidence (test AUROC ${f3(M[DET].auroc)} against ${f3(M.heuristic_confidence.auroc)}); and (iv) an exploratory study showing that self-consistency aggregation raises accuracy from ${pc(single)} to ${pc(agg)} on validation, and that the detector is most useful as a router that obtains the same accuracy with about ${most.mean_vlm_calls.toFixed(1)} VLM calls per question in place of four.`),

  H1('2. Review of existing techniques'),
  P('**Hallucination benchmarks.** CHAIR [4] measures objects mentioned in captions but absent from the image. POPE [5] recasts this as Yes/No polling (“Is there a ... in the image?”), which removes the need to parse free text and exposes a strong bias towards answering “Yes”. R-Bench [6] extends polling to relationships and reports that relationship hallucinations are more frequent than object ones. For spatial relations specifically, Visual Spatial Reasoning (VSR) [2] provides human-validated true/false statements, and What’sUp [7] shows that VLMs often fail to distinguish a relation from its opposite. We follow the polling format and build on Visual Genome [1], whose scene graphs supply subject–predicate–object annotations at scale.'),
  P('**Confidence and selective prediction.** The simplest error signal is the model’s own output probability, but modern networks are often miscalibrated [8]. Selective classification [9] formalises the trade-off between coverage (the share of inputs answered) and risk (the error rate on those answered); Whitehead et al. [10] apply it to visual question answering and argue that abstaining is preferable to answering incorrectly. Our abstention experiments use the same risk–coverage view, with thresholds fixed on validation data.'),
  P('**Probing internal states.** Linear probes [11] test whether a property can be read from a hidden layer. For language models, Azaria and Mitchell [12] train a small classifier on hidden activations to predict whether a statement is true, and Burns et al. [13] find truth-related directions without labels. These works suggest that a model’s internal state can be more informative than its output. Our detectors apply this idea to a VLM and to the model’s own answer, and compare a linear probe with a one-hidden-layer MLP on three depths.'),
  P('**Consistency and cost-aware inference.** Self-consistency [14] samples several answers and takes the most common one; SelfCheckGPT [15] flags hallucinations by disagreement between samples. Both need several model calls. Cascades such as FrugalGPT [16] reduce cost by calling an expensive model only when a cheap scorer is unsure. Our aggregation uses logically related rephrasings of one fact, not repeated samples, and our cascade uses the hidden-state detector as the scorer.'),
  P('**Gap addressed.** Prior work mostly evaluates either a detector or a mitigation strategy. We connect them on one task: the same detector is measured as a classifier (RQ1, RQ2) and then as a component in three ways of acting on its output (RQ3), with compute cost reported alongside accuracy.'),

  H1('3. Methods'),
  H2('3.1 Task and data'),
  P(`Visual Genome version 1.4 contains ${int(filters.filter_counts.relationships_total)} relationship annotations. We keep relationships whose predicate, after lower-casing, exactly matches a surface form of one of eight canonical spatial predicates: *to the left of*, *to the right of*, *above*, *below*, *in front of*, *behind*, *on* and *under*. Vague predicates such as “near” or “next to” are excluded because they have no opposite. After removing duplicates and relationships whose subject and object share a name, ${int(filters.kept_relationships)} relationships in ${int(filters.kept_images)} images remain. We further drop ${int(filters.label_conflicts_total)} relationships whose negated statement is itself annotated as true in the same image, since their label would be contradictory.`),
  P('Visual Genome only annotates relations that hold. To obtain balanced labels, each relationship (subject S, predicate, object O) produces four statements: the **original** (“The S is on the O.”, true), the **converse** (“The O is under the S.”, true), the **swapped** statement (“The O is on the S.”, false) and the **opposite** statement (“The S is under the O.”, false). Every relationship therefore contributes two true and two false questions about the same underlying fact, which also makes consistency measurable.'),
  P(`Images are assigned to train, validation and test in proportions 70/15/15 by a seeded hash of the image identifier, so no image or relationship appears in two splits. We sample 250 relationships per predicate (at most two per image), giving ${int(meta.counts.train.relationships + meta.counts.val.relationships + meta.counts.test.relationships)} relationships and ${int(meta.counts.train.rows + meta.counts.val.rows + meta.counts.test.rows)} questions: ${int(meta.counts.train.rows)} train, ${int(meta.counts.val.rows)} validation and ${int(meta.counts.test.rows)} test. Images that also occur in the VSR subset used in Section 4.6 are excluded (${filters.vg_images_also_in_teammate_vsr} images).`),
  H2('3.2 Frozen VLM and extracted features'),
  P('The VLM is Qwen2.5-VL-3B-Instruct at a pinned revision, loaded with 4-bit NF4 quantisation [17] (vision tower and output head left unquantised), bfloat16 computation, batch size 1 and at most 200,704 image pixels. It is never fine-tuned. Each question uses the prompt *Statement: “...” Is this statement true in the image? Answer Yes or No.* One forward pass yields, at the position where the answer begins, the logits of the “Yes” and “No” tokens and the hidden state of dimension 2,048 at layers 9, 18 and 36 (early, middle and final of 36).'),
  P('From the two logits we define p_yes = softmax(Yes, No) and confidence = max(p_yes, 1 − p_yes). The VLM answers Yes when p_yes ≥ 0.5. The detector target is y_error = 1 when this answer disagrees with the label. Truth labels, the VLM’s answer, identifiers and split are never given to a detector.'),
  H2('3.3 Error detectors'),
  P('Each detector maps an input to a risk score in [0, 1], where higher means the answer is more likely wrong. We compare four input sets, each used separately: the two confidence features [p_yes, confidence], and the hidden state at layer 9, 18 or 36. Inputs are standardised with statistics from the training split.'),
  B('**Baseline.** risk = 1 − confidence, with nothing fitted.'),
  B('**Logistic regression (LR).** L2 penalty, LBFGS; C in {0.001, 0.01, 0.1, 1, 10}; class weights none or balanced.'),
  B('**MLP.** Linear(d, 32) → ReLU → Dropout(0.2) → Linear(32, 1) with a sigmoid output; binary cross-entropy; Adam with learning rate 0.001; batch size 32; at most 200 epochs with early stopping (patience 15) on validation loss; weight decay in {0.0001, 0.01}; class weights none or balanced; seeds 42, 43 and 44.'),
  P(`Hyperparameters are chosen on validation AUROC (ties by average precision); for the MLP the mean over the three seeds is used and seed 42 is declared the primary run in advance. In total ${frozen.candidates_tried.lr} LR fits and ${frozen.candidates_tried.mlp_seed_fits} MLP fits were made. All fitted models, thresholds and file hashes were then frozen, and the test split was scored exactly once.`),
  H2('3.4 Evaluation'),
  P(`We report AUROC, average precision (AP; its chance level is the error rate) and the Brier score. Confidence intervals are 95% percentile intervals from ${int(results.protocol.bootstrap_repetitions)} bootstrap resamples of test images [18], so that the several questions of one image stay together; differences between two detectors use the same resamples (paired). For abstention, a threshold is chosen on validation so that at most 80% of questions are answered, and is applied unchanged to test.`),
  H2('3.5 Three ways to use the signals'),
  B('**Abstention.** Answer only when the risk is below the threshold. This costs one VLM call and can only withhold an answer.'),
  B('**Aggregation.** Ask all four variants of a fact. Each call gives a support for the fact: p_yes for the original and converse, 1 − p_yes for the swapped and opposite statements. The mean of the four supports decides all four answers (fact true if the mean is at least 0.5). This is a confidence-weighted vote, not a majority vote, and costs four calls. The distance of the mean from 0.5 (the margin) can be used to abstain.'),
  B('**Cascade (detect, then verify).** Every question first gets one call. A router ranks the answers by risk; only the riskiest share receives the three additional calls and the aggregated answer. Sweeping the share from 0 to 1 traces accuracy against mean calls per question. A random router gives the straight line between the two end points, so any gain above that line is due to the router.'),
  P('We also test three aggregation rules that use the detector’s error probability for each of the four calls: weighting each support by 1 − P(wrong); replacing the confidence by detector votes; and averaging both. These analyses were designed after the test evaluation, so they use the validation split only and are reported as exploratory.'),

  H1('4. Experiments and results'),
  H2('4.1 Set-up'),
  P(`Experiments ran on one NVIDIA RTX 4060 Laptop GPU (8 GB) with Python 3.12, PyTorch ${frozen.versions.torch}, Transformers 4.57.6 and scikit-learn ${frozen.versions['scikit-learn']}. Extracting all ${int(extraction.queries)} questions took ${(extraction.wall_seconds / 60).toFixed(0)} minutes (${extraction.wall_seconds_per_new_query.toFixed(2)} s per question, peak ${extraction.peak_allocated_vram_gib.toFixed(2)} GiB of GPU memory); training and selecting all detectors took ${(frozen.training_wall_seconds / 60).toFixed(0)} minutes. The answer was always one of the two expected tokens.`),
  H2('4.2 How often, and where, the VLM is wrong'),
  P(`On the test split the VLM answers ${pc(test.unfiltered_vlm_accuracy)} of ${int(test.rows)} questions correctly (${test.errors} errors in ${test.images} images). Figure 1 shows that errors are not uniform. The hardest predicate is *${worstPred}* (${pc(byPred[worstPred].vlm_accuracy)}), close to guessing, and the easiest is *${bestPred}* (${pc(byPred[bestPred].vlm_accuracy)}). The model is also inconsistent about a single fact: it accepts ${pc(byVar.orig.vlm_accuracy)} of original statements but only ${pc(byVar.converse.vlm_accuracy)} of their converses, which state the same thing from the other object’s point of view. It says “Yes” to ${pc(byVar.swap.vlm_yes_rate)} of swapped and ${pc(byVar.opposite.vlm_yes_rate)} of opposite statements, both false: these are relation hallucinations in the strict sense.`),
  figure('figure_vlm_accuracy.png', 640, 209),
  caption('Figure 1. Accuracy of the frozen VLM on the test split, by canonical predicate (left) and by statement variant (right). The dashed line is the overall accuracy.'),
  H2('4.3 Detecting wrong answers (RQ1, RQ2)'),
  P('Table 1 and Figure 2 (left) give the held-out test results.'),
  table(['Detector input', 'LR: AUROC [95% CI]', 'LR: AP', 'MLP: AUROC [95% CI]', 'MLP: AP'], detectionRows, [30, 24, 8, 24, 8]),
  caption(`Table 1. Error detection on the test split (${int(test.rows)} questions, error rate ${f3(test.error_prevalence)}, which is the chance level for AP). MLP results are for the primary seed 42.`),
  B(`**The final layer beats confidence.** The layer-${LAYER} MLP reaches AUROC ${f3(M[DET].auroc)}, against ${f3(M.mlp_all_confidence.auroc)} for the MLP on confidence; the paired difference is ${sg(mlpVsConf.auroc_delta)} ${sci(mlpVsConf.paired_image_bootstrap95)}. For LR the difference is ${sg(lrVsConf.auroc_delta)} ${sci(lrVsConf.paired_image_bootstrap95)}. Both intervals exclude zero.`),
  B(`**Fitting on confidence adds nothing.** LR and MLP on [p_yes, confidence] are indistinguishable from the unfitted baseline (${f3(M.heuristic_confidence.auroc)}); every paired interval includes zero.`),
  B('**Early and middle layers are worse than confidence.** Layers 9 and 18 give AUROC between ' + `${f3(Math.min(M.lr_all_hidden9.auroc, M.mlp_all_hidden9.auroc, M.lr_all_hidden18.auroc, M.mlp_all_hidden18.auroc))} and ${f3(Math.max(M.lr_all_hidden9.auroc, M.mlp_all_hidden9.auroc, M.lr_all_hidden18.auroc, M.mlp_all_hidden18.auroc))}` + ', significantly below the baseline (Appendix C).'),
  B(`**The MLP helps a little at layer ${LAYER}.** It is ahead of LR by ${sg(mlpVsLr.auroc_delta)} ${sci(mlpVsLr.paired_image_bootstrap95)}; over three seeds its test AUROC is ${f3(test.seed_stability[DET].test_auroc_mean)} ± ${f3(test.seed_stability[DET].test_auroc_std)}. At layers 9 and 18 the MLP is not better than LR.`),
  figure('figure_test_results.png', 640, 279),
  caption('Figure 2. Left: test AUROC with 95% image-bootstrap intervals for each input set (blue: logistic regression; orange: MLP; dotted line: unfitted confidence baseline). Right: risk–coverage curves on test; dots mark the thresholds frozen on validation.'),
  H2('4.4 Abstention'),
  table(['Risk score', 'Coverage', 'Accuracy on answered', 'Errors withheld', 'Precision of abstention'], abstainRows, [30, 14, 20, 18, 20]),
  caption('Table 2. Abstention on test at the threshold frozen on validation (target: at most 80% coverage).'),
  P(`Abstention raises accuracy on the answered questions from ${pc(test.unfiltered_vlm_accuracy)} to about 75% (Table 2). At this operating point the hidden-state detector is only slightly better than plain confidence (${pc(op(DET).retained_vlm_accuracy)} against ${pc(op('heuristic_confidence').retained_vlm_accuracy)}), although its withheld answers are more often truly wrong (${pc(op(DET).abstention_precision)} against ${pc(op('heuristic_confidence').abstention_precision)}). The risk–coverage curves in Figure 2 (right) show that its advantage grows as more answers are withheld: on validation, at 60% coverage, it gives ${pc(at('abstain_detector', 0.6).accuracy_on_answered)} against ${pc(at('abstain_confidence', 0.6).accuracy_on_answered)} for confidence (Table 3).`),
  H2('4.5 Abstention, aggregation and the cascade (RQ3; validation, exploratory)'),
  table(['Strategy', 'VLM calls', 'Accuracy, all answered', 'About 80% coverage', 'About 60% coverage'], usageRows, [34, 10, 16, 20, 20]),
  caption(`Table 3. Three ways of using the signals, on the validation split (${int(extra.rows)} questions, ${extra.relationships} relationships). Cells show accuracy on answered questions @ actual coverage; tied scores are kept together, so coverage can fall below the target.`),
  P(`Aggregation is the largest single improvement: accuracy rises from ${pc(single)} to ${pc(agg)}, a gain of ${pts(gain.accuracy_gain)} points (95% interval [${pts(gain.image_bootstrap95[0])}, ${pts(gain.image_bootstrap95[1])}]). The reason is the inconsistency seen in Section 4.2: the single-query VLM answers all four variants correctly for only ${pc(cons.all_four_correct)} of relationships, so averaging corrects the cases where one phrasing went wrong. A plain majority vote would not work, because ${pc(cons.two_correct)} of relationships split two against two; the confidence weighting breaks these ties. Strict hallucinations (“Yes” to a false statement) fall from ${pc(err.single_false_yes_rate)} to ${pc(err.aggregate_false_yes_rate)}. Adding abstention on the aggregate margin reaches ${pc(at('aggregate_abstain_margin', 0.8).accuracy_on_answered)} at 80% coverage.`),
  P('Aggregation costs four calls per question. The cascade asks how much of its benefit can be kept for less (Table 4, Figure 3).'),
  table(['Mean VLM calls', 'Share verified', 'Random router', 'Router: confidence', `Router: LR layer ${LAYER}`, `Router: MLP layer ${LAYER}`, 'MLP router + abstain to 80%'], cascadeRows, [11, 11, 12, 14, 14, 14, 16]),
  caption('Table 4. Cascade on validation: accuracy with all questions answered, by the share of questions sent for verification. The last column additionally withholds the least decided verified questions until coverage is 80%.'),
  figure('figure_cascade.png', 400, 288),
  caption('Figure 3. Cascade accuracy against mean VLM calls per question (validation). The dashed line is a router that picks questions at random.'),
  B(`**The detector is an effective router.** Verifying half of the questions (${half.mean_vlm_calls.toFixed(1)} calls) gives ${pc(half.accuracy)} with the MLP router, against ${pc(half.random_router_accuracy)} at random; the gain over random is [${pts(half.gain_over_random_image_bootstrap95[0])}, ${pts(half.gain_over_random_image_bootstrap95[1])}] points.`),
  B(`**Full accuracy at lower cost.** At ${most.mean_vlm_calls.toFixed(1)} calls the MLP router has sent ${pc(most.single_errors_escalated)} of the wrong single answers for verification and matches always-verify (${pc(most.accuracy)} against ${pc(agg)}).`),
  B(`**The hidden state matters here.** At ${half.mean_vlm_calls.toFixed(1)} calls the confidence router gives ${pc(route('confidence', 0.5).accuracy)}, below the MLP router. Below about 1.6 calls the routers are indistinguishable.`),
  H2('4.6 Using the detector inside the aggregation'),
  table(['Aggregation rule', 'Accuracy', 'Change vs confidence only [95% CI], points', 'Yes to a false statement', 'Abstain to 80%', 'Cascade at 2.5 calls'], ruleRows, [30, 11, 25, 13, 11, 11]),
  caption(`Table 5. Aggregation rules that also use the layer-${LAYER} MLP’s error probability for each of the four calls (validation).`),
  P(`None of the three detector-informed rules is reliably better than plain confidence averaging (Table 5): trust weighting changes accuracy by ${pts(tw.minus_confidence_rule)} points and every interval includes zero. Replacing confidence by detector votes is the weakest rule (${pc(dv.accuracy)}). Once four related answers are available, their agreement already carries most of what the detector knows; the detector is therefore more valuable for deciding *whether* to verify than for *how* to combine the verification.`),
  H2('4.7 A second dataset: VSR'),
  P(`To check that the findings are not specific to our generated questions, the same frozen VLM configuration and the same detector grids were run on the above, below, left-of and right-of statements of VSR [2] (941 human-validated statements in 855 images, split by image; ${v('heuristic_confidence').rows} test questions). Table 6 places the two test evaluations side by side. They use different data, so the columns are not directly comparable.`),
  table(['Quantity (test split)', 'VSR', 'Visual Genome (this work)'], vsrRows, [40, 30, 30]),
  caption('Table 6. The two test evaluations side by side. Abstention cells show accuracy on answered questions @ coverage, at thresholds frozen on validation.'),
  P(`The pattern is the same on both: confidence gives an AUROC of about 0.7, only the final layer is useful, and layers 9 and 18 are close to guessing. On VSR the layer-36 MLP (${f3(v('mlp_all_hidden36').auroc)}) does not separate from the confidence detectors, which is expected with only ${v('heuristic_confidence').errors} test errors: its interval is about three times as wide as ours. VSR also starts from a higher accuracy, and its labels are human-validated, whereas Visual Genome labels are noisier. [[Team: add Jeremy’s results here if they are to be included (one short paragraph and at most one table).]]`),

  H1('5. Discussion'),
  P(`**What we learned.** (1) A small detector on the final hidden layer predicts the VLM’s spatial errors better than the VLM’s own confidence, by about ${f3(mlpVsHeur.auroc_delta)} AUROC over the unfitted baseline. The information is therefore present in the network but partly lost when the state is reduced to a Yes/No probability. (2) This information appears late. Probes on layers 9 and 18 overfit (Appendix B) and generalise worse than confidence, so depth matters more than detector capacity; the MLP’s advantage over a linear probe is small. (3) The dominant failure is inconsistency: the VLM gives different answers to logically equivalent or contradictory statements about one fact. Exploiting this through aggregation removes far more errors than any detector-based abstention. (4) The detector and aggregation are complementary. The detector costs almost nothing once the forward pass is done, and its best use is to decide which answers deserve the expensive cross-check.`),
  P('**Practical trade-off.** With one call per question, the detector can only withhold answers, and gains a few points of accuracy on those it keeps. With about 2.5 calls on average, the cascade recovers most of the aggregation gain; with about 3.1 calls it recovers all of it. Always aggregating is simplest and needs no access to hidden states, but spends calls on answers that were already right.'),
  P('**Limitations.** (i) Truth labels come from Visual Genome annotations without relabelling or a bounding-box geometry check, so some “errors” are label noise; this lowers every AUROC and is a likely reason for the weaker abstention gains than on VSR. (ii) Converse statements built from a predicate’s opposite (“the table is under the cup”) are not always natural English, and an image may contain several objects with the same name. (iii) Sections 4.5 and 4.6 were designed after the test evaluation and use the validation split, on which the detector hyperparameters were also selected; their numbers are therefore slightly optimistic and need confirmation on fresh data. (iv) Aggregation requires rewriting a question into its converse, swapped and opposite forms. Here these come from a template; for free-form questions the rewriting itself would be a source of error. (v) Validation chose the strongest regularisation in the planned LR grid, so stronger settings were not explored. (vi) We study one model, one quantisation setting and one prompt, and the VLM is never modified: we detect and filter hallucinations but do not remove their cause.'),
  P('**Future work.** The most direct next steps are a geometry-based label filter using Visual Genome bounding boxes, cross-evaluation of each dataset’s detectors on the other, a two-call cascade (a statement and its opposite only), and confirmation of the cascade on a fresh held-out split or an external benchmark such as R-Bench [6].'),

  H1('6. AI Use Statement'),
  P('We used Claude Code (Anthropic), an AI coding assistant, throughout this project. Its use was as follows.'),
  B('**Code.** The assistant wrote most of the pipeline code (data preparation, VLM feature extraction, detector training, evaluation, figures and unit tests) under our direction, following a written team plan. We reviewed the code and ran it on our own hardware.'),
  B('**Analysis.** The assistant ran the experiments and produced the result tables. Every number in this report is generated by a script from saved results; none was typed by hand or produced by the assistant from memory.'),
  B('**Writing.** The assistant produced the first draft of this report from the saved results. We edited the text and are responsible for its content. The references were suggested by the assistant and [[each reference must be checked against the original source by a team member before submission]].'),
  B('**Decisions made by us.** The research questions, the choice of model and datasets, the detector grids, the decision to drop contradictory labels, the dataset size, and the three usage strategies (abstention, aggregation, cascade, and using the detector inside the aggregation) were specified by team members.'),
  P('The conversation transcripts are included in the code submission (folder `claude/`). [[Team: add each member’s own AI use here, including any tools other than Claude Code.]]'),
];

const REFERENCES = [
  'R. Krishna et al., “Visual Genome: Connecting language and vision using crowdsourced dense image annotations,” International Journal of Computer Vision, vol. 123, no. 1, pp. 32–73, 2017.',
  'F. Liu, G. Emerson and N. Collier, “Visual Spatial Reasoning,” Transactions of the Association for Computational Linguistics, vol. 11, pp. 635–651, 2023.',
  'S. Bai et al., “Qwen2.5-VL Technical Report,” arXiv:2502.13923, 2025.',
  'A. Rohrbach, L. A. Hendricks, K. Burns, T. Darrell and K. Saenko, “Object hallucination in image captioning,” in Proc. EMNLP, 2018.',
  'Y. Li, Y. Du, K. Zhou, J. Wang, W. X. Zhao and J.-R. Wen, “Evaluating object hallucination in large vision-language models,” in Proc. EMNLP, 2023.',
  'M. Wu et al., “Evaluating and analyzing relationship hallucinations in large vision-language models,” in Proc. ICML, 2024.',
  'A. Kamath, J. Hessel and K.-W. Chang, “What’s ‘up’ with vision-language models? Investigating their struggle with spatial reasoning,” in Proc. EMNLP, 2023.',
  'C. Guo, G. Pleiss, Y. Sun and K. Q. Weinberger, “On calibration of modern neural networks,” in Proc. ICML, 2017.',
  'Y. Geifman and R. El-Yaniv, “Selective classification for deep neural networks,” in Proc. NeurIPS, 2017.',
  'S. Whitehead, S. Petryk, V. Shakib, J. Gonzalez, T. Darrell, A. Rohrbach and M. Rohrbach, “Reliable visual question answering: Abstain rather than answer incorrectly,” in Proc. ECCV, 2022.',
  'G. Alain and Y. Bengio, “Understanding intermediate layers using linear classifier probes,” in ICLR Workshop, 2017.',
  'A. Azaria and T. Mitchell, “The internal state of an LLM knows when it’s lying,” in Findings of EMNLP, 2023.',
  'C. Burns, H. Ye, D. Klein and J. Steinhardt, “Discovering latent knowledge in language models without supervision,” in Proc. ICLR, 2023.',
  'X. Wang, J. Wei, D. Schuurmans, Q. Le, E. Chi, S. Narang, A. Chowdhery and D. Zhou, “Self-consistency improves chain of thought reasoning in language models,” in Proc. ICLR, 2023.',
  'P. Manakul, A. Liusie and M. J. F. Gales, “SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models,” in Proc. EMNLP, 2023.',
  'L. Chen, M. Zaharia and J. Zou, “FrugalGPT: How to use large language models while reducing cost and improving performance,” arXiv:2305.05176, 2023.',
  'T. Dettmers, A. Pagnoni, A. Holtzman and L. Zettlemoyer, “QLoRA: Efficient finetuning of quantized LLMs,” in Proc. NeurIPS, 2023.',
  'B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. Chapman & Hall, 1993.',
];
const references = [pageBreak(), H1('References'),
  ...REFERENCES.map((r, i) => new Paragraph({ children: runs(`[${i + 1}] ${r}`), spacing: { after: 60 },
    indent: { left: 400, hanging: 400 } }))];

const COMMANDS = ['uv venv --python 3.12 .venv',
  'uv pip install --python .venv/bin/python -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match',
  '.venv/bin/python scripts/vlm.py                           # smoke test of the VLM',
  '.venv/bin/python -I scripts/01_download_metadata.py      # Visual Genome metadata (official links)',
  '.venv/bin/python -I scripts/02_parse_filter.py           # predicate mapping and filters',
  '.venv/bin/python -I scripts/03_split_sample.py --per-predicate 250',
  '.venv/bin/python -I scripts/04_download_images.py        # only the images needed',
  '.venv/bin/python -I scripts/05_extract.py                # VLM forward passes (about 45 minutes)',
  '.venv/bin/python -I scripts/06_build_npz.py              # detector-ready arrays',
  '.venv/bin/python -I scripts/07_train_detectors.py        # train, select on validation, freeze',
  '.venv/bin/python -I scripts/08_evaluate_test.py          # single test evaluation',
  '.venv/bin/python -I scripts/10_aggregate.py              # abstention, aggregation, cascade (validation)',
  '.venv/bin/python -I scripts/09_report.py                 # tables and figures',
  '.venv/bin/python -m pytest tests -q                      # unit tests'];
const appendix = [pageBreak(), H1('Appendix A. Reproducing the results'),
  P('Requirements: Linux, an NVIDIA GPU with at least 6 GB of memory, Python 3.12 and about 10 GB of disk. Dependencies are pinned in `requirements.txt` (torch 2.9.1+cu126, transformers 4.57.6, accelerate 1.12.0, bitsandbytes 0.49.2, scikit-learn, pandas, matplotlib). Run from the repository root, in this order. All sampling uses seeded hashes (seed 42), so the same questions are produced on every run. `README.md` describes each file and `ASSUMPTIONS.md` records every design decision.'),
  ...COMMANDS.map((c) => new Paragraph({ children: [new TextRun({ text: c, font: 'Courier New', size: 16 })], spacing: { after: 20 } })),
  H1('Appendix B. Selected hyperparameters'),
  table(['Model', 'Selected setting', 'Class weights', 'Validation AUROC', 'Test AUROC', 'Test AUROC, 3 seeds'], selectedRows, [26, 18, 13, 14, 12, 17]),
  caption('Table B1. Settings selected on validation and the resulting AUROCs (MLP: primary seed 42; last column mean ± standard deviation over seeds 42, 43 and 44). For hidden layers 9 and 18, training AUROC was far above validation AUROC for every weakly regularised LR setting (for example 0.925 against 0.529 at layer 9 with C = 10), which is why the strongest regularisation was selected.'),
  H1('Appendix C. Paired AUROC comparisons'),
  table(['Comparison (test)', 'AUROC difference', '95% paired interval', 'Interval'], comparisonRows, [48, 16, 20, 16]),
  caption('Table C1. All paired comparisons on the test split (same image-bootstrap resamples for both models). Exploratory; not adjusted for multiplicity.'),
  H1('Appendix D. Breakdowns by predicate and variant'),
  table(['Predicate', 'VLM accuracy (test)', 'Yes rate (test)', 'AUROC: 1 − confidence', `AUROC: LR layer ${LAYER}`, `AUROC: MLP layer ${LAYER}`, 'Single-query accuracy (val)', 'Aggregated accuracy (val)'], predicateRows, [16, 12, 10, 12, 12, 12, 13, 13]),
  caption('Table D1. Per-predicate results: VLM behaviour and detector AUROC on test (152 questions per predicate), and the effect of aggregation on validation (152 questions per predicate).'),
  table(['Variant', 'VLM accuracy', 'Yes rate', 'AUROC: 1 − confidence', 'AUROC: LR confidence', `AUROC: LR layer ${LAYER}`, `AUROC: MLP layer ${LAYER}`], variantRows, [16, 14, 12, 15, 15, 14, 14]),
  caption('Table D2. Per-variant results on test (304 questions per variant). Within one variant the truth label is constant, so p_yes alone ranks errors; this is why the fitted confidence detector differs from the unfitted baseline here.'),
];

const heading = (size, before) => ({ run: { font: FONT, size, bold: true, color: '000000' },
  paragraph: { spacing: { before, after: 100 }, keepNext: true } });
const doc = new Document({
  creator: 'SC4001 project group',
  title: 'Detecting and Reducing Spatial-Relation Errors of a Frozen Vision Language Model',
  styles: { default: { document: { run: { font: FONT, size: 20 } },
    heading1: heading(26, 240), heading2: heading(22, 160) } },
  numbering: { config: [{ reference: 'bullets', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•',
    alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 400, hanging: 240 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT] })] })] }) },
    children: [...cover, ...contents, ...body, ...references, ...appendix],
  }],
});

const target = path.join(OUT, 'SC4001_ProjectE_Report_DRAFT.docx');
Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(target, buffer);
  const words = body.length;
  console.log(`Wrote ${path.relative(ROOT, target)} (${(buffer.length / 1024).toFixed(0)} KB, ${words} body blocks)`);
});
