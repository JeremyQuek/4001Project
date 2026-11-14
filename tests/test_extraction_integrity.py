"""Regression checks for resumable writes and hidden-state alignment."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from extract_vsr_features import atomic_json,validate_hidden


class ExtractionIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='vsr_integrity_')
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'state.pt'

    def save_hidden(self,shape=2048,nan=False,position='final prompt token, before answering'):
        vectors={layer:torch.ones(shape) for layer in (9,18,36)}
        if nan:
            vectors[18][0]=float('nan')
        torch.save({'position':position,'hidden_state_indices':[9,18,36],'vectors':vectors},self.path)

    def test_transient_windows_lock_is_retried(self):
        target=Path(self.temp.name)/'progress.json'
        original=Path.replace
        attempts=[]
        def flaky(path,destination):
            attempts.append(1)
            if len(attempts)==1:
                raise PermissionError('Simulated Windows reader lock')
            return original(path,destination)
        with patch.object(Path,'replace',new=flaky),patch('extract_vsr_features.time.sleep'):
            atomic_json(target,{'complete':7})
        self.assertEqual(json.loads(target.read_text()),{'complete':7})
        self.assertEqual(len(attempts),2)

    def test_valid_hidden_states(self):
        self.save_hidden()
        validate_hidden(self.path)

    def test_wrong_hidden_shape_rejected(self):
        self.save_hidden(shape=2047)
        with self.assertRaises(ValueError):
            validate_hidden(self.path)

    def test_nonfinite_hidden_rejected(self):
        self.save_hidden(nan=True)
        with self.assertRaises(ValueError):
            validate_hidden(self.path)

    def test_wrong_token_alignment_rejected(self):
        self.save_hidden(position='after generated answer')
        with self.assertRaises(ValueError):
            validate_hidden(self.path)


if __name__=='__main__':
    unittest.main()
