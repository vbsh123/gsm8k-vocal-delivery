from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import audit_reasoning_emotion as audit
import collect_phi4_reasoning_emotion as phi
import collect_diffa_reasoning_emotion as diffa


class GeometryAuditTests(unittest.TestCase):
    def test_noncollinear_signed_projection_known_answer(self):
        r = torch.tensor([3., 4.], dtype=torch.float64)
        e = torch.tensor([-4., 0.], dtype=torch.float64)
        g = audit.geometry(r, e)
        self.assertAlmostEqual(g['cosine'], -.6)
        self.assertAlmostEqual(g['coefficient'], -.48)
        self.assertAlmostEqual(g['overlap'], .36)
        self.assertLess(g['orthogonality'], 1e-15)
        self.assertLess(g['pythagorean_error'], 1e-15)
        self.assertAlmostEqual(audit.geometry(r*10, e)['coefficient'], -.048)
        self.assertAlmostEqual(audit.geometry(r*10, e)['overlap'], .36)
        self.assertIsNone(audit.geometry(torch.zeros(2), e)['cosine'])

    def fixture(self, root, model):
        groups = ['acoustic', 'semantic', 'combined', 'answer'] if model == 'diffa' else ['answer', 'audio']
        shape = (32, len(groups), 3)
        pairs = [dict(question_id=f'q{i}', voice='Jenny',
                      neutral=dict(audio_sha256=f'n{i}'), angry=dict(audio_sha256=f'a{i}')) for i in range(100)]
        plan = dict(protocol=audit.PROTOCOLS[model], pairs=pairs, groups=groups,
                    layers=list(range(32)), conditions=list(audit.CONDITIONS), reasoning_prompt='reason',
                    answer_only_prompt='direct', caveat='test')
        (root/'run_config.json').write_text(json.dumps(plan)); (root/'states').mkdir()
        r = torch.tensor([3., 4., 0.]).expand(shape).clone()
        e = torch.tensor([-4., 0., 0.]).expand(shape).clone()
        for i, pair in enumerate(pairs):
            baseline = torch.full(shape, float(i))
            for c, h in zip(audit.CONDITIONS, (baseline, baseline-r, baseline+e)):
                meta = dict(protocol=plan['protocol'], question_id=pair['question_id'], voice='Jenny', condition=c,
                            prompt='direct' if c == 'neutral-answer-only' else 'reason',
                            audio_sha256=pair['angry' if c == 'angry-reasoning' else 'neutral']['audio_sha256'],
                            input_tokens=530, audio_tokens=4)
                if model == 'diffa':
                    meta.update(states=h, answer_tokens=512, answer_input_sha256='mask',
                                speech_bounds=dict(start=2, acoustic_end=4, end=6))
                else:
                    meta.update(answer=h[:, 0], audio=h[:, 1])
                torch.save(meta, root/'states'/f'{i:03d}-{c}.pt')
        with redirect_stdout(io.StringIO()):
            (diffa if model == 'diffa' else phi).summarize(root)
        return shape

    def test_both_collectors_match_independent_raw_state_math(self):
        for model in ('diffa', 'phi4'):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as d:
                root = Path(d); shape = self.fixture(root, model)
                before = (root/'layer-cosines.csv').read_bytes()
                with patch.object(audit, 'state_shape', return_value=shape):
                    result = audit.audit(root, model)
                self.assertEqual(result['status'], 'PASS', result['errors'])
                self.assertEqual(before, (root/'layer-cosines.csv').read_bytes())
                self.assertAlmostEqual(result['checks'][0]['overlap'], .36)

    def test_changed_prompt_cache_and_nonfinite_states_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); shape = self.fixture(root, 'phi4')
            path = root/'states'/'000-neutral-reasoning.pt'
            saved = torch.load(path, weights_only=True); saved['prompt'] = 'wrong'
            torch.save(saved, path)
            with patch.object(audit, 'state_shape', return_value=shape):
                with self.assertRaisesRegex(ValueError, 'metadata'):
                    audit.audit(root, 'phi4')
                saved['prompt'] = 'reason'; saved['answer'][0, 0] = float('nan'); torch.save(saved, path)
                with self.assertRaisesRegex(ValueError, 'nonfinite'):
                    audit.audit(root, 'phi4')

    def test_csv_sign_error_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); shape = self.fixture(root, 'phi4')
            path = root/'layer-cosines.csv'
            with path.open() as stream: rows = list(csv.DictReader(stream))
            rows[0]['signed_projection_coefficient'] = '0.48'
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            with patch.object(audit, 'state_shape', return_value=shape): result = audit.audit(root, 'phi4')
            self.assertEqual(result['status'], 'FAIL')
            self.assertTrue(any('signed_projection_coefficient' in e for e in result['errors']))


if __name__ == '__main__': unittest.main()
