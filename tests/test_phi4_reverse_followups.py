from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import phi4_reverse_followups as run
import steer_phi4_gsm_reverse as engine


class Native:
    def __init__(self, layer=17): self.layer = layer; self.calls = []
    def generate(self, row, prompt, direction=None, strength=0., observe=False, observe_layers=None):
        self.calls.append((row['question_id'], row['style'], strength, observe))
        result = dict(response='#### 2', response_tokens=4, hook_forwards=2)
        if observe:
            sign = 1. if row['style'] == 'angry' else 0.
            result['mean'] = torch.full((3072,), sign*(self.layer+1))
            if observe_layers is not None:
                result['means'] = {i: torch.full((3072,), sign*(i+1)) for i in observe_layers}
        return result


def source_fixture(root, total=1269):
    source = root/'audio'; source.mkdir()
    questions = [dict(question_id=f'q{i:04d}', question='Question', answer='2') for i in range(total)]
    records = [dict(question_id=q['question_id'], voice='Jenny', style=s, condition=s,
                    audio_file=f'/audio/{q["question_id"]}-{s}.wav', audio_sha256=q['question_id']+s,
                    reference_answer='2', seconds=40.) for q in questions for s in ('neutral', 'angry')]
    return source, questions, dict(recipe=dict(engine.MODELS['phi4']), recordings=records)


class ReverseFollowupTests(unittest.TestCase):
    def test_l30_same_split_and_reuse_l17_neutral_baselines(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root = Path(d); source, questions, parent = source_fixture(root)
            plans = []
            with patch.object(engine.gsm, 'questions', return_value=questions), patch.object(engine.gsm, 'make_plan', return_value=parent):
                for layer in (17, 30):
                    out = root/f'L{layer}'; out.mkdir()
                    plan = engine.prepare(SimpleNamespace(source=source, output=out, layer=layer))
                    plans.append((out, plan))
            out17, p17 = plans[0]; out30, p30 = plans[1]
            self.assertEqual(p17['collection'], p30['collection'])
            self.assertEqual(p17['test'], p30['test'])
            self.assertEqual(p30['layer'], 30)
            self.assertEqual(list(p30['strengths'].values()), [0, -.5, -1])
            h17 = engine.digest(out17/'run_config.json'); h30 = engine.digest(out30/'run_config.json')
            native = Native(17); v = engine.collect(p17, out17, native, h17); engine.evaluate(p17, out17, native, v, h17)
            before = (out17/'results.jsonl').read_bytes()
            self.assertEqual(engine.reuse_answers(p30, out30, h30, out17, baseline_only=True), 400)
            rows, done = engine.completed(p30, out30, h30)
            self.assertEqual(len(rows), 400); self.assertTrue(all(r['condition']=='neutral' for r in rows))
            self.assertTrue(all(r['layer']==30 and r['reused_evidence']['original_layer']==17 for r in rows))
            self.assertEqual(engine.reuse_answers(p30, out30, h30, out17, baseline_only=True), 0)
            self.assertEqual(before, (out17/'results.jsonl').read_bytes())

    def test_gsm_remaining_469_keep_frozen_vector_and_original_answers(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root = Path(d); source, questions, parent = source_fixture(root); old_out = root/'old'; old_out.mkdir()
            new_out = root/'new'; new_out.mkdir()
            with patch.object(engine.gsm, 'questions', return_value=questions), patch.object(engine.gsm, 'make_plan', return_value=parent):
                old = engine.prepare(SimpleNamespace(source=source, output=old_out, layer=30))
                h = engine.digest(old_out/'run_config.json'); native = Native(30)
                v = engine.collect(old, old_out, native, h); engine.evaluate(old, old_out, native, v, h)
                source_bytes = (old_out/'results.jsonl').read_bytes()
                plan = run.prepare_gsm(source, old_out, new_out, 30)
                self.assertEqual(len(plan['test']), 869)
                self.assertEqual(plan['test'][:400], old['test'])
                self.assertFalse({p['question_id'] for p in plan['collection']}&set(plan['bank']))
                torch.testing.assert_close(run.install_fixed_vector(plan, new_out), v)
                new_hash = engine.digest(new_out/'run_config.json')
                self.assertEqual(engine.reuse_answers(plan, new_out, new_hash, old_out), 1200)
                calls_before = len(native.calls)
                engine.evaluate(plan, new_out, native, v, new_hash)
                self.assertEqual(len(native.calls)-calls_before, 469*3)
                rows, _ = engine.completed(plan, new_out, new_hash)
                self.assertEqual(len(rows), 2607)
                self.assertEqual(source_bytes, (old_out/'results.jsonl').read_bytes())
                self.assertEqual(plan, run.prepare_gsm(source, old_out, new_out, 30))
                extra_out = root/'extra'; extra_out.mkdir()
                retained = (new_out/'results.jsonl').read_bytes()
                extra = run.prepare_gsm_extra(new_out, extra_out, 30, -1.5)
                self.assertEqual(extra['test'], plan['test'])
                self.assertEqual(list(extra['strengths'].values()), [0., -1.5])
                torch.testing.assert_close(run.install_fixed_vector(extra, extra_out), v)
                extra_hash = engine.digest(extra_out/'run_config.json')
                self.assertEqual(engine.reuse_answers(extra, extra_out, extra_hash, new_out), 869)
                calls_before = len(native.calls)
                engine.evaluate(extra, extra_out, native, v, extra_hash)
                self.assertEqual(len(native.calls)-calls_before, 869)
                self.assertEqual(retained, (new_out/'results.jsonl').read_bytes())
                with (old_out/'results.jsonl').open('ab') as stream: stream.write(b'changed')
                with self.assertRaisesRegex(ValueError, 'source evidence changed'): run.prepare_gsm(source, old_out, new_out, 30)

    def test_multiarith_collect_once_two_layers_own290_split(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root = Path(d); source, questions, raw = source_fixture(root, total=580)
            records = {(r['question_id'], r['style']): r for r in raw['recordings']}
            pairs = [dict(question_id=q['question_id'], voice='Jenny', neutral=records[q['question_id'],'neutral'],
                          angry=records[q['question_id'],'angry']) for q in questions[:290]]
            test = [records[q['question_id'],'neutral'] for q in questions[290:]]
            bank = {q['question_id']:q for q in questions}; shared = root/'shared'; out17=root/'L17'; out30=root/'L30'
            out17.mkdir(); out30.mkdir()
            with patch.object(run, 'multi_selection', return_value=(pairs,test,bank)):
                p17 = run.prepare_multi(source, out17, 17, shared)
                p30 = run.prepare_multi(source, out30, 30, shared)
            h17=engine.digest(out17/'run_config.json');h30=engine.digest(out30/'run_config.json')
            native=Native(17)
            a=run.collect_multi(p17,out17,native,h17)
            self.assertEqual(len(native.calls),580)
            b=run.collect_multi(p30,out30,native,h30)
            self.assertEqual(len(native.calls),580)  # No duplicated collection generations.
            torch.testing.assert_close(a,torch.full((3072,),18.))
            torch.testing.assert_close(b,torch.full((3072,),31.))
            engine.evaluate(p17,out17,native,a,h17)
            self.assertEqual(engine.reuse_answers(p30,out30,h30,out17,baseline_only=True),290)
            calls_before=len(native.calls)
            engine.evaluate(p30,out30,native,b,h30)
            self.assertEqual(len(native.calls)-calls_before,580)  # Only two steered conditions, baseline retained.
            rows,_=engine.completed(p30,out30,h30)
            self.assertEqual(len(rows),870)


if __name__=='__main__':unittest.main()
