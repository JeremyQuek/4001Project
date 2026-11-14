"""Check preprocessing boundaries, risk orientation and attainable coverage."""
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_logistic_baselines import coverage_threshold,fit_candidate,load_training_split


class LogisticTrainingTests(unittest.TestCase):
    def test_scaler_fitted_on_training_only_and_risk_is_error_class(self):
        X=np.array([[-3.,0],[-2.,1],[-1.,0],[1.,1],[2.,0],[3.,1]])
        y=np.array([0,0,0,1,1,1])
        config={'solver':'lbfgs','max_iter':5000,'tol':1e-6,'seed':42}
        model=fit_candidate(X,y,1,None,config)
        self.assertEqual(model['scaler'].n_samples_seen_,len(X))
        np.testing.assert_allclose(model['scaler'].mean_,X.mean(axis=0))
        before=model['scaler'].mean_.copy()
        p=model.predict_proba(np.array([[-100.,0],[100.,0]]))[:,1]
        self.assertLess(p[0],p[1])
        np.testing.assert_array_equal(before,model['scaler'].mean_)
        with tempfile.TemporaryDirectory(prefix='lr_roundtrip_') as temporary:
            path=Path(temporary)/'model.joblib'
            joblib.dump(model,path)
            np.testing.assert_allclose(joblib.load(path).predict_proba(X),model.predict_proba(X),rtol=0,atol=1e-12)

    def test_test_split_cannot_be_opened_by_training_stage(self):
        with self.assertRaises(ValueError):
            load_training_split('test')

    def test_coverage_respects_ties_and_target(self):
        y=np.array([0,0,1,0,1])
        risk=np.array([.1,.2,.2,.2,.9])
        result=coverage_threshold(y,risk,.6)
        self.assertEqual(result['answered'],1)
        self.assertLessEqual(result['coverage'],.6)

    def test_identical_scores_cannot_force_partial_tie_acceptance(self):
        result=coverage_threshold(np.array([0,1,0,1]),np.full(4,.5),.8)
        self.assertEqual(result['answered'],0)
        self.assertIsNone(result['retained_vlm_accuracy'])

    def test_retained_accuracy_uses_original_error_labels(self):
        result=coverage_threshold(np.array([0,1,0,1,1]),np.arange(5)/5,.6)
        self.assertEqual(result['answered'],3)
        self.assertEqual(result['retained_errors'],1)
        self.assertAlmostEqual(result['retained_vlm_accuracy'],2/3)


if __name__=='__main__':
    unittest.main()
