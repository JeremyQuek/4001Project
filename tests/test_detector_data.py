"""Test cohort alignment, feature boundaries, and artifact integrity without fitting."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from load_detector_data import load_detector_data


class DetectorDataTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='vsr_array_test_')
        self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)
        self.core=np.array([[.2,.8,.1,.1],[.8,.8,.1,.7],[.4,.6,.2,.2]],dtype=np.float32)
        behavior=np.full((3,7),np.nan,dtype=np.float32)
        behavior[:,:4]=self.core
        behavior[[0,2],4:]=[[.1,.2,.3],[.4,.5,.6]]
        self.target=np.array([1,0,1],dtype=np.int8)
        path=self.folder/'train.npz'
        np.savez_compressed(path,example_ids=np.array(['a','b','c']),image_ids=np.array(['1','2','3']),
            relations=np.array(['above','left of','below']),y_error=self.target,
            consistency_mask=np.array([True,False,True]),truth_label=np.array([1,0,1]),
            X_confidence=self.core[:,:2],X_core=self.core,X_behavior=behavior,
            **{f'H_layer{layer}':np.full((3,2048),layer,dtype=np.float32) for layer in (9,18,36)})
        (self.folder/'metadata.json').write_text(json.dumps({'ready_for_training':True,
            'checksums':{'train.npz':hashlib.sha256(path.read_bytes()).hexdigest()}}))

    def load(self,**kwargs):
        return load_detector_data(folder=self.folder,**kwargs)

    def test_core_does_not_include_labels_or_identifiers(self):
        X,y,info=self.load()
        np.testing.assert_array_equal(X,self.core)
        np.testing.assert_array_equal(y,self.target)
        self.assertEqual(info['example_ids'].tolist(),['a','b','c'])

    def test_behavior_requires_matched_cohort(self):
        with self.assertRaises(ValueError):
            self.load(feature_set='behavior')

    def test_cohort_keeps_targets_and_comparators_aligned(self):
        X,y,info=self.load(feature_set='behavior',cohort='consistency')
        baseline,baseline_y,baseline_info=self.load(feature_set='confidence',cohort='consistency')
        self.assertEqual(X.shape,(2,7))
        self.assertEqual(baseline.shape,(2,2))
        np.testing.assert_array_equal(y,baseline_y)
        np.testing.assert_array_equal(info['example_ids'],baseline_info['example_ids'])
        np.testing.assert_array_equal(y,[1,1])

    def test_hidden_layer_selection(self):
        X,_,_=self.load(feature_set='hidden',layer=36)
        self.assertEqual(X.shape,(3,2048))
        self.assertTrue((X==36).all())

    def test_combined_dimensions(self):
        X,_,_=self.load(feature_set='combined_core')
        Y,_,_=self.load(feature_set='combined_behavior',cohort='consistency')
        self.assertEqual(X.shape,(3,2052))
        self.assertEqual(Y.shape,(2,2055))

    def test_modified_artifact_rejected(self):
        with (self.folder/'train.npz').open('ab') as handle:
            handle.write(b'changed')
        with self.assertRaises(ValueError):
            self.load()

    def test_unknown_split_and_layer_rejected(self):
        for kwargs in ({'split':'unknown'},{'layer':0},{'cohort':'unknown'}):
            with self.assertRaises(ValueError):
                self.load(**kwargs)


if __name__=='__main__':
    unittest.main()
