"""MLP architecture, deterministic scoring, and checkpoint integrity checks."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_mlp_detectors import ErrorMLP,fit_seed,load_checkpoint,risk_scores


class MLPTests(unittest.TestCase):
    def test_two_learned_layers_and_output_shape(self):
        model=ErrorMLP(4)
        self.assertEqual(sum(isinstance(m,torch.nn.Linear) for m in model.modules()),2)
        self.assertEqual(tuple(model(torch.ones(5,4)).shape),(5,))

    def test_evaluation_disables_dropout(self):
        model=ErrorMLP(4)
        X=np.ones((5,4),dtype=np.float32)
        a=risk_scores(model,X)
        b=risk_scores(model,X)
        np.testing.assert_array_equal(a,b)
        self.assertTrue(((a>=0)&(a<=1)).all())

    def test_saved_model_reproduces_risk(self):
        model=ErrorMLP(4)
        X=np.ones((5,4),dtype=np.float32)
        with tempfile.TemporaryDirectory(prefix='mlp_test_') as folder:
            path=Path(folder)/'model.pt'
            torch.save({'state_dict':model.state_dict(),'input_dim':4,'hidden_units':32,'dropout':.2},path)
            loaded,_=load_checkpoint(path)
            np.testing.assert_array_equal(risk_scores(loaded,X),risk_scores(model,X))

    def test_training_keeps_best_validation_checkpoint_and_training_weights(self):
        torch.set_num_threads(1)
        X=np.array([[-2.],[-1.],[-.5],[.5],[1.],[2.]],dtype=np.float32)
        y=np.array([0,0,0,0,1,1])
        config={'hidden_units':4,'dropout':.2,'learning_rate':.01,'batch_size':3,
            'max_epochs':12,'patience':3,'min_delta':1e-5}
        result=fit_seed(X,y,X,y,config,42,.01,'balanced')
        self.assertEqual(result['positive_weight'],2.)
        self.assertLessEqual(result['best_epoch'],result['epochs_run'])
        self.assertTrue(np.isfinite(result['validation_risk']).all())
        self.assertFalse(result['model'].training)


if __name__=='__main__':
    unittest.main()
