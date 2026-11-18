from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import evaluate_gsm_happy400 as run
import steer_minicpm_gsm as mini


class HappyTests(unittest.TestCase):
    def test_fixed400_reuses_only_matching_neutral_cache(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root = Path(d); source = root/'source'; source.mkdir()
            questions = [dict(question_id=f'q{i}', question='Q', answer='2') for i in range(1269)]
            selected = questions[:400]
            (source/'questions.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in selected))
            records = {(q['question_id'], s): dict(question_id=q['question_id'], voice='Jenny', style=s,
                answer='2', audio_file='audio', audio_sha256=q['question_id']+s) for q in selected for s in ('neutral', 'happy')}
            args = SimpleNamespace(model='minicpm', source=source, output=root/'out')
            with patch.object(run.gsm, 'questions', return_value=questions), patch.object(run.reasoning, 'source_records', return_value=records):
                output, plan = run.prepare(args)
            self.assertEqual(len(plan['recordings']), 800)
            self.assertEqual(plan['styles'], ['neutral', 'happy'])
            self.assertEqual(set(plan['bank']), {q['question_id'] for q in selected})
            self.assertEqual(plan['prompt'], run.gsm.ORIGINAL)
            old_root = root/'outputs/minicpm-gsm400-869-L30-a1'; (old_root/'pairs').mkdir(parents=True)
            neutral = dict(plan['recordings'][0]); angry = dict(neutral, style='angry', audio_sha256='q0angry')
            pair = dict(question_id='q0', voice='Jenny', neutral=neutral, angry=angry)
            options = dict(enable_thinking=False, use_tts_template=True, generate_audio=False, do_sample=False,
                max_new_tokens=512, min_new_tokens=0, num_beams=1, repetition_penalty=1., max_inp_length=32768, stream=False)
            old = dict(protocol=mini.PROTOCOL, prompt=plan['prompt'], recipe=plan['recipe'], seed=42,
                max_new_tokens=512, native_options=options, collection=[pair])
            config = old_root/'run_config.json'; config.write_text(json.dumps(old))
            traces = {s: dict(mean=torch.full((4096,), float(i)), response='#### 2', response_tokens=4, hook_forwards=2)
                      for i, s in enumerate(('neutral', 'angry'))}
            saved = dict(protocol=mini.PROTOCOL, run_config_sha256=run.digest(config), question_id='q0', voice='Jenny',
                audio_sha256={'neutral': 'q0neutral', 'angry': 'q0angry'}, traces=traces, answer_difference=torch.ones(4096))
            torch.save(saved, old_root/'pairs/0000.pt')
            import os
            previous = Path.cwd()
            try:
                os.chdir(root)
                run.reuse_collection(output, plan); run.reuse_collection(output, plan)
                rows = run.read_rows(output/'results.jsonl')
                self.assertEqual(len(rows), 1); self.assertEqual(rows[0]['style'], 'neutral')
                self.assertEqual(rows[0]['response'], '#### 2'); self.assertEqual(rows[0]['interventions'], [])
                happy = dict(plan['recordings'][1], response='#### 3', response_tokens=4, prompt=plan['prompt'])
                with (output/'results.jsonl').open('a') as stream: stream.write(json.dumps(happy)+'\n')
                predictions = ['2', '3']
                values = {m: [dict(prediction=p, correct=p=='2') for p in predictions] for m in ('strict-match', 'flexible-extract')}
                scorer = SimpleNamespace(score=lambda responses, docs: values, provenance={'test': True})
                with patch('rescore_audio_steering.HarnessTaskScorer', return_value=scorer): run.gsm.report(output)
                summary = json.loads((output/'summary.json').read_text())
                self.assertEqual([a['style'] for a in summary['accuracy']], ['neutral', 'happy'])
                self.assertEqual(summary['paired'][0]['losses'], 1)
                old['prompt'] = 'Changed'; config.write_text(json.dumps(old))
                with self.assertRaisesRegex(ValueError, 'prompt/model/decoder'): run.reuse_collection(output, plan)
            finally: os.chdir(previous)


if __name__ == '__main__': unittest.main()
