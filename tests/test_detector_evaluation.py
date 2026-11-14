import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_detector_comparison import operating_point, paired_bootstrap, risk_coverage


class EvaluationTests(unittest.TestCase):
    def test_threshold_keeps_whole_ties_without_targeting_test_coverage(self):
        result=operating_point(np.array([0,1,1,0]),np.array([.2,.2,.7,.8]),.2)
        self.assertEqual(result['answered'],2)
        self.assertEqual(result['retained_errors'],1)
        self.assertEqual(result['retained_vlm_accuracy'],.5)
        self.assertEqual(result['error_recall'],.5)

    def test_no_answers_has_no_accuracy(self):
        result=operating_point(np.array([0,1]),np.array([.2,.8]),.1)
        self.assertIsNone(result['retained_vlm_accuracy'])
        self.assertEqual(result['answered'],0)

    def test_risk_curve_keeps_ties_and_reaches_full_sample_risk(self):
        coverage,risk=risk_coverage(np.array([0,1,1,0]),np.array([.2,.2,.7,.8]))
        np.testing.assert_allclose(coverage,[.5,.75,1])
        np.testing.assert_allclose(risk,[.5,2/3,.5])

    def test_identical_models_have_zero_paired_difference(self):
        y=np.array([0,1,0,1,0,1])
        risk=np.array([.1,.4,.2,.9,.3,.8])
        result=paired_bootstrap(y,{'a':risk,'b':risk.copy()},np.array(['x','x','y','y','z','z']),
            {'same':('a','b')},30)
        np.testing.assert_array_equal(result['comparisons']['same']['paired_image_bootstrap95'],[0,0])
        self.assertEqual(result['valid_resamples'],30)


if __name__=='__main__':
    unittest.main()
