from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import run_reasoning_transfer as run


def rows(path, values):
    path.write_text(''.join(json.dumps(v)+'\n' for v in values))


def wav(path, seconds=1):
    with wave.open(str(path), 'wb') as audio:
        audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(100)
        audio.writeframes(b'\0\0'*int(100*seconds))


class PreparationTests(unittest.TestCase):
    def test_queue_runs_stages_sequentially_and_stops_on_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
                root = Path(d); source = root/'source'; source.mkdir()
                (source/'SVAMP-source.json').write_bytes(b'source')
                (source/'questions.jsonl').write_text('')
                (source/'plan.jsonl').write_text('')
                output = root/'out'; stages = []
                def child(command, **kw):
                    stage = command[-1]; stages.append(stage)
                    self.assertTrue(kw['check'])
                    if fail and stage == 'svamp':
                        raise run.subprocess.CalledProcessError(1, command)
                with patch.object(sys, 'argv', ['run_reasoning_transfer.py', '--output', str(output),
                        '--svamp-existing', str(source), '--gsm-audio-sources', str(source)]), \
                     patch.object(run, 'digest', return_value='hash'), patch.object(run, 'svamp_split'), \
                     patch.object(run, 'gsm_sources', return_value=([], root/'vector')), \
                     patch.object(run.subprocess, 'run', side_effect=child):
                    if fail:
                        with self.assertRaises(run.subprocess.CalledProcessError):
                            run.main()
                    else:
                        run.main()
                self.assertEqual(stages, ['bbh5', 'svamp'] if fail else ['bbh5', 'svamp', 'gsm8k'])
                self.assertEqual(json.loads((output/'status.json').read_text()),
                    dict(stage='svamp', status='failed') if fail else dict(stage='finished', status='complete'))

    def test_five_object_options_and_prompt_are_aligned_with_frozen_vector_transfer(self):
        original = dict(input='Five birds.\nOptions:\n'+'\n'.join(f'({c}) Bird {c}' for c in 'ABCDE'), target='(E)')
        questions = run.bbh_questions(json.dumps(dict(examples=[original]*250)))
        self.assertEqual(len({q['question_id'] for q in questions}), 250)
        self.assertIn('Option E: Bird E', questions[0]['question'])
        self.assertNotIn('The answer is', questions[0]['question'])
        self.assertEqual(questions[0]['original_target'], '(E)')
        self.assertIn('five choices A, B, C, D, E', run.BBH_PROMPT)
        self.assertIn('The answer is (<letter>).', run.BBH_PROMPT)
        self.assertNotIn('three choices', run.BBH_PROMPT)

    def test_svamp_existing_500_unchanged_and_remaining_500_are_disjoint(self):
        data = json.dumps([dict(ID=str(i), Body=f'Body {i}.', Question='How many?',
            Answer=i, Equation=f'{i}+0', Type='Addition') for i in range(1000)]).encode()
        existing = run.svamp_questions(data, 500)
        before = json.dumps(existing)
        with patch.object(run, 'SVAMP_SHA', hashlib.sha256(data).hexdigest()):
            train, test = run.svamp_split(data, existing)
            self.assertEqual(train, existing)
            self.assertEqual((len(train), len(test)), (500, 500))
            self.assertFalse({q['question_id'] for q in train} & {q['question_id'] for q in test})
            changed = [dict(q) for q in existing]; changed[0]['answer'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'selection is unchanged'):
                run.svamp_split(data, changed)
        self.assertEqual(json.dumps(existing), before)

    def test_tts_new_svamp_is_neutral_only_and_resumes(self):
        with tempfile.TemporaryDirectory() as d, patch.object(run.subprocess, 'run') as command:
            root = Path(d)
            run.generate_audio(root, [dict(question_id='q', question='How many?', answer='5')], ['neutral'])
            argv = command.call_args.args[0]
            self.assertEqual(argv[argv.index('--styles')+1:argv.index('--output')], ['neutral'])
            self.assertIn('--resume', argv)
            self.assertNotIn('answer', run.read_rows(root/'questions.jsonl')[0]['question'])

    def test_gsm_exact_718_preserved_with_long_angry_and_new_prompt(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); output = root/'out'; output.mkdir()
            source = root/'azure'; source.mkdir()
            wav(source/'neutral.wav'); wav(source/'angry.wav', 40)
            vector = root/'vector.pt'; vector.write_bytes(b'frozen vector')
            bank = [dict(question=f'Math question {i}', answer='Reasoning\n#### 5') for i in range(718)]
            rows(output/'gsm8k-test.jsonl', bank)
            records = [dict(question_id=f'gsm8k_test_{i:05d}', speaker='Jenny',
                audio_path=str(source/'neutral.wav'), answer='5') for i in range(718)]
            cohorts = []
            for name, subset in [('second', records[:460]), ('third', records[460:])]:
                folder = root/name; folder.mkdir(); cohorts.append(folder)
                run.freeze(folder/'run_config.json', dict(recordings=subset, settings=dict(vectors=str(vector))))
            rows(source/'plan.jsonl', [dict(question_id=r['question_id'], voice='Jenny', provider='azure',
                style=s, audio_path=s+'.wav', question=bank[i]['question'], answer='5')
                for i, r in enumerate(records) for s in ('neutral', 'angry')])
            args = SimpleNamespace(gsm_cohorts=cohorts, gsm_vector=None, gsm_audio_sources=[source])
            plan = run.prepare_gsm(args, output)
            self.assertEqual(len(plan['bank']), 718)
            self.assertEqual(len(plan['recordings']), 2154)
            self.assertEqual(plan['extraction'], [])
            self.assertEqual({r['condition'] for r in plan['recordings']}, {'neutral', 'angry', run.STEERED})
            self.assertEqual(plan['prompt'], run.HASH_PROMPT)
            self.assertEqual(plan['collection_prompt'], run.GSM_COLLECTION_PROMPT)
            self.assertTrue(all('#### <number>' in plan['prompt'] for _ in [0]))
            self.assertEqual(sum(r['seconds'] > 30 for r in plan['recordings']), 718)
            self.assertEqual(vector.read_bytes(), b'frozen vector')

    def test_changed_prompt_cannot_resume_original_result_folder(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'config.json'
            run.freeze(path, dict(prompt='old'))
            with self.assertRaisesRegex(ValueError, 'settings changed'):
                run.freeze(path, dict(prompt='new'))


class EvaluationTests(unittest.TestCase):
    def test_real_hooks_collect_matched_means_then_steer_only_neutral_with_seed42(self):
        import numpy as np
        import torch
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            def task(name, qid, style, condition=None):
                file = root/name; file.write_bytes(name.encode())
                return dict(question_id=qid, voice='Jenny', style=style, answer='10', audio_file=str(file),
                    audio_sha256=run.digest(file), condition=condition or style)
            pairs = [dict(question_id=f'train{i}',
                neutral=task(f'n{i}', f'train{i}', 'neutral'), angry=task(f'a{i}', f'train{i}', 'angry')) for i in range(2)]
            base = task('test', 'heldout', 'neutral')
            plan = dict(task='svamp', extraction=pairs, prompt=run.HASH_PROMPT, collection_prompt=run.HASH_PROMPT,
                vector=str(root/'style-vectors.pt'), bank=dict(heldout=dict(answer='10')),
                recordings=[dict(base, condition=c) for c in ('neutral', run.STEERED)])
            class Block(torch.nn.Module):
                def forward(self, h):
                    return (h,)
            blocks = torch.nn.ModuleList([Block() for _ in range(32)])
            model = SimpleNamespace(llm_model=SimpleNamespace(model=SimpleNamespace(transformer=SimpleNamespace(blocks=blocks))))
            numbers = {'n0': [1., 3.], 'a0': [5., 9., 999.], 'n1': [2.], 'a1': [11., 999.], 'test': [10.]}
            calls = []
            def gen(audio, sr, received, tokenizer, features, prompt, tokens, seconds):
                self.assertIs(received, model)
                self.assertEqual((sr, prompt, tokens, seconds, torch.initial_seed()), (16000, run.HASH_PROMPT, 512, 30, 42))
                for x in numbers[audio]:
                    h = torch.full((1, 514, 1), x)
                    final = blocks[31](h)[0]
                    self.assertEqual(final[0, 0, 0], x)
                calls.append(audio)
                return f'#### {final[0, -1, 0].item():g}', dict(mode='native', chunk_seconds=30)
            class Scorer:
                def score(self, response, gold):
                    return {'flexible-numeric': response == '#### '+gold}, {'flexible-extract': response.split()[-1]}
            original_to = torch.Tensor.to
            def cpu_to(t, *a, **kw):
                if kw.get('device') == 'cuda':
                    kw['device'] = 'cpu'
                return original_to(t, *a, **kw)
            with patch.dict(sys.modules, {'soundfile': SimpleNamespace(read=lambda file, **kw: (Path(file).name, 16000)),
                    'extract_vectors': SimpleNamespace(gen_chunked=gen, load_model=lambda: (model, None, None))}), \
                 patch.object(run, 'scorer_for', return_value=Scorer()), patch.object(torch.Tensor, 'to', cpu_to), \
                 patch.dict(os.environ, {'HF_HOME': str(root/'cache')}):
                run.evaluate(plan, root)
                # Resume does not generate any completed answer or repeat collection.
                run.evaluate(plan, root)
            self.assertEqual(calls, ['n0', 'a0', 'n1', 'a1', 'test', 'test'])
            state = torch.load(root/'style-vectors.pt', weights_only=True)
            self.assertEqual(state['pair_counts']['angry'][31], 2)
            self.assertEqual(state['pair_mean_sums']['angry'][31, 3].item()/2, 7.)
            saved = run.read_rows(root/'results.jsonl')
            self.assertEqual([r['response'] for r in saved], ['#### 10', '#### 17'])
            self.assertEqual([r['steered_forwards'] for r in saved], [0, 1])
            self.assertFalse(blocks[31]._forward_hooks)

    def test_summary_can_be_refreshed_and_uses_paired_completed_questions(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root = Path(d)
            plan = dict(task='svamp', bank={'q1': dict(answer='5'), 'q2': dict(answer='5')},
                recordings=[dict(question_id=q, voice='Jenny', condition=c) for q in ('q1', 'q2') for c in ('neutral', run.STEERED)])
            run.freeze(root/'run_config.json', plan)
            class Scorer:
                provenance = dict(test=True)
                def score(self, response, gold):
                    return {'flexible-numeric': response == gold}, {'flexible-extract': response}
            initial = [dict(question_id='q1', voice='Jenny', condition=c, reference_answer='5', response=r)
                       for c, r in [('neutral', '4'), (run.STEERED, '5')]]
            rows(root/'results.jsonl', initial)
            with patch.object(run, 'scorer_for', return_value=Scorer()):
                run.report(root)
                rows(root/'results.jsonl', initial+[dict(question_id='q2', voice='Jenny', condition='neutral', reference_answer='5', response='5')])
                run.report(root)
            summary = json.loads((root/'summary.json').read_text())
            self.assertEqual(summary['paired'][0]['pairs'], 1)
            self.assertEqual((summary['paired'][0]['wins'], summary['paired'][0]['losses']), (1, 0))
            self.assertEqual(len(run.read_rows(root/'rescored.jsonl')), 3)


@unittest.skipUnless(os.environ.get('REASONING_HARNESS_CACHE'), 'Set REASONING_HARNESS_CACHE for actual pinned upstream scoring')
class ActualHarnessTests(unittest.TestCase):
    def test_numeric_official_string_metrics_separate_from_numeric_equivalence(self):
        scorer = run.NumericScorer(Path(os.environ['REASONING_HARNESS_CACHE'])/'numeric')
        self.assertTrue(scorer.score('#### 5', '5')[0]['strict-match'])
        self.assertFalse(scorer.score('The result is 5', '5')[0]['strict-match'])
        self.assertTrue(scorer.score('The result is 5', '5')[0]['flexible-extract'])
        values, _ = scorer.score('#### 5.00', '5')
        self.assertFalse(values['flexible-extract']); self.assertTrue(values['flexible-numeric'])

    def test_five_object_harness_accepts_all_choices(self):
        scorer = run.HarnessScorer(Path(os.environ['REASONING_HARNESS_CACHE'])/'bbh5', objects='five')
        docs = [dict(input='Which?\nOptions:\n'+'\n'.join(f'({c}) bird {c}' for c in 'ABCDE'), target=f'({letter})') for letter in 'ABCDE']
        values = scorer.score([f'The answer is ({letter}).' for letter in 'ABCDE'], docs)
        self.assertTrue(all(v['correct'] for group in values.values() for v in group))


if __name__ == '__main__':
    unittest.main()
