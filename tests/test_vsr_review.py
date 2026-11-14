"""Review persistence and extraction-gate tests, using disposable fixtures only."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from serve_vsr_review import ReviewError, ReviewStore


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='vsr_review_test_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        images = {}
        for i,split in enumerate(('train','validation','test','train'),1):
            image_id = f'{i:012}'
            path = self.root/f'data/vsr/images/{image_id}.jpg'
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(b'disposable image fixture '+bytes([i]))
            images[image_id] = {'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            self.rows.append({'example_id':f'example-{i}','image_id':image_id,
                'statement':f'Fixture statement {i}.','truth_label':i%2,'split':split,
                'variants':{'converse_truth':None,'mirrored_equivalent_truth':None,'exclusive_truth':None}})
        source = self.root/'data/vsr/processed/directional_candidates.jsonl'
        source.parent.mkdir(parents=True,exist_ok=True)
        source.write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        dataset = {'dataset_fingerprint':'fixture-dataset','rows':self.rows,'images':images,
                   'candidate_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
        self.folder = self.root/'data/vsr/review'
        self.folder.mkdir(parents=True)
        (self.folder/'dataset.json').write_text(json.dumps(dataset))
        self.store = ReviewStore(self.root)

    def payload(self,**kwargs):
        return {'dataset_fingerprint':'fixture-dataset','expected_revision':self.store.state['revision'],**kwargs}

    def decide(self,index,status='approve',reasons=None,note=''):
        return self.store.save(self.payload(example_id=f'example-{index}',
            decision={'status':status,'reasons':reasons or [],'note':note}))

    def complete(self):
        self.store.save(self.payload(reviewer='Test reviewer'))
        self.decide(1)
        self.decide(2,'original_only',['reference_frame'])
        self.decide(3)
        self.decide(4,'set_aside',['label_questionable'])

    def test_initial_state_is_unreviewed(self):
        self.assertEqual(self.store.counts()['unreviewed'],4)
        self.assertFalse(self.store.path.exists())

    def test_decisions_persist(self):
        self.decide(1)
        restored = ReviewStore(self.root)
        self.assertEqual(restored.state['decisions']['example-1']['status'],'approve')

    def test_stale_revision_is_rejected(self):
        payload = self.payload(reviewer='Old tab')
        self.decide(1)
        with self.assertRaises(ReviewError):
            self.store.save(payload)

    def test_reasons_required(self):
        for status in ('original_only','set_aside'):
            with self.assertRaises(ReviewError):
                self.decide(1,status)

    def test_approval_cannot_keep_issue_flags(self):
        with self.assertRaises(ReviewError):
            self.decide(1,'approve',['label_questionable'])

    def test_original_only_requires_usable_original(self):
        with self.assertRaises(ReviewError):
            self.decide(1,'original_only',['objects_unclear'])

    def test_other_requires_explanation(self):
        with self.assertRaises(ReviewError):
            self.decide(1,'set_aside',['other'])

    def test_unknown_statement_rejected(self):
        with self.assertRaises(ReviewError):
            self.decide(100)

    def test_incomplete_review_cannot_greenlight(self):
        self.store.save(self.payload(reviewer='Test reviewer'))
        self.decide(1)
        with self.assertRaises(ReviewError):
            self.store.greenlight(self.payload(confirm=True))

    def test_confirmation_required(self):
        self.complete()
        with self.assertRaises(ReviewError):
            self.store.greenlight(self.payload(confirm=False))

    def test_greenlight_preserves_labels_and_separates_cohorts(self):
        self.complete()
        result = self.store.greenlight(self.payload(confirm=True))
        approval = result['greenlight']
        self.assertEqual(approval['artifacts']['extraction_manifest']['rows'],3)
        self.assertEqual(approval['artifacts']['consistency_manifest']['rows'],2)
        self.assertEqual(approval['artifacts']['set_aside']['rows'],1)
        for artifact in approval['artifacts'].values():
            self.assertEqual(hashlib.sha256((self.root/artifact['path']).read_bytes()).hexdigest(),artifact['sha256'])
        all_path = self.root/approval['artifacts']['reviewed_all']['path']
        rows = [json.loads(x) for x in all_path.read_text().splitlines()]
        self.assertEqual([r['truth_label'] for r in rows],[r['truth_label'] for r in self.rows])
        self.assertIsNone(rows[1]['variants']['converse_truth'])
        self.assertEqual(rows[0]['variants']['converse_truth'],rows[0]['truth_label'])
        self.assertTrue(all(r['variants']['exclusive_truth'] is None for r in rows))

    def test_later_edit_invalidates_approval(self):
        self.complete()
        self.store.greenlight(self.payload(confirm=True))
        self.decide(1,'unreviewed')
        self.assertIsNone(self.store.state['greenlight'])

    def test_changed_image_blocks_greenlight(self):
        self.complete()
        (self.root/'data/vsr/images/000000000001.jpg').write_bytes(b'changed')
        with self.assertRaises(ReviewError):
            self.store.greenlight(self.payload(confirm=True))

    def test_changed_source_blocks_greenlight(self):
        self.complete()
        with (self.root/'data/vsr/processed/directional_candidates.jsonl').open('a') as handle:
            handle.write('\n')
        with self.assertRaises(ReviewError):
            self.store.greenlight(self.payload(confirm=True))

    def test_each_split_must_have_included_examples(self):
        self.complete()
        self.decide(2,'set_aside',['label_questionable'])
        with self.assertRaises(ReviewError):
            self.store.greenlight(self.payload(confirm=True))


if __name__=='__main__':
    unittest.main()
