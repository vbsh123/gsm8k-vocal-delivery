from contextlib import redirect_stdout
import io
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np
import torch

import winning_diffa_sweeps as run


class WinnerSweepTests(unittest.TestCase):
    def fixture(self, root, name='gsm8k', task='gsm8k'):
        folder = root/name; folder.mkdir()
        legacy = task in ('bbh3', 'ifeval')
        tokens = 1024 if task == 'ifeval' else 512
        qid = name+'_test'
        audio = folder/'neutral.wav'
        with wave.open(str(audio), 'wb') as stream:
            stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(16000)
            stream.writeframes(b'\0\0'*1600)
        answer = 'A' if task.startswith('bbh') else ('' if task == 'ifeval' else '10')
        bank = {qid: dict(question_id=qid, question='question', answer=answer,
            original_input='Which?\nOptions:\n(A) Red\n(B) Blue\n(C) Green', original_target='(A)',
            key=1, prompt='Write at least two words.', instruction_id_list=[], kwargs=[])}
        record = dict(question_id=qid, voice='Jenny', style='neutral', condition='neutral', answer=answer,
                      audio_file=str(audio), audio_path=audio.name, audio_sha256=run.digest(audio))
        config = dict(task=task, prompt='target-'+name, seed=42, max_new_tokens=tokens,
            decoder='fast_dllm', steps=128, block_length=32, temperature=0., layer=31, group='answer', strength=1,
            extraction=[{'question_id': name+'_train'}])
        if legacy:
            config['test'] = [dict(question_id=qid, neutral=record)]
            if task == 'ifeval':
                from run_spoken_ifeval import REVISION
                config['revision'] = REVISION
            (folder/'questions.jsonl').write_text(json.dumps(bank[qid])+'\n')
        else:
            config.update(bank=bank, recordings=[record], collection_prompt='collection-'+name,
                          vector=str(folder/'style-vectors.pt'))
        (folder/('split.json' if legacy else 'run_config.json')).write_text(json.dumps(config))
        sums = torch.zeros(32, 1, 3); sums[31, 0] = 2.
        counts = torch.zeros(32, dtype=torch.long); counts[31] = 1
        torch.save(dict(pair_mean_sums={'angry': sums}, pair_counts={'angry': counts}, groups=['answer'],
            mode='fixed', method='mean', prompt=config.get('collection_prompt', config['prompt']),
            max_new_tokens=tokens, extraction_question_ids=[name+'_train']), folder/'style-vectors.pt')
        base = dict(record, prompt=config['prompt'], seed=42, max_new_tokens=tokens, decoder='fast_dllm',
                    reference_answer=answer, response='The answer is (A).' if task.startswith('bbh') else '#### 10')
        if task == 'ifeval':
            base.pop('prompt'); base['wrapper'] = config['prompt']
        positive = dict(base, condition=run.OWN_CONDITIONS.get(task, 'neutral-L31-answer-a1'))
        if legacy:
            base.pop('audio_sha256'); positive.pop('audio_sha256')
        else:
            positive['interventions'] = [dict(layer=31, group='answer', strength=1.)]
        (folder/'results.jsonl').write_text(json.dumps(base)+'\n'+json.dumps(positive)+'\n')
        return dict(task=task, run=str(folder), **({'source': str(folder)} if legacy else {}))

    def test_exact_grid_and_off_diagonal_matrix(self):
        self.assertEqual(run.STRENGTHS, (-2., -1.5, -1., -.5, 0., .5, 1., 1.5, 2.))
        datasets = dict.fromkeys(['gsm8k', 'svamp', 'multiarith', 'ifeval', 'bbh3', 'bbh5'])
        own = run.jobs_for(datasets, 'strengths'); transfer = run.jobs_for(datasets, 'transfer')
        self.assertEqual((len(own), len(transfer)), (6, 30))
        self.assertTrue(all(j['source'] == j['target'] and j['strengths'] == list(run.STRENGTHS) for j in own))
        self.assertTrue(all(j['source'] != j['target'] and j['strengths'] == [0., 1.] for j in transfer))

    def test_all_six_formats_load_and_keep_ifeval_budget_and_native_legacy_reuse(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root = Path(d)
            entries = {name: self.fixture(root, name, name) for name in ['gsm8k', 'svamp', 'multiarith', 'ifeval', 'bbh3', 'bbh5']}
            catalog = root/'catalog.json'; catalog.write_text(json.dumps(entries))
            output = root/'sweep'; output.mkdir()
            manifest = run.prepare(catalog, output, 'strengths')
            self.assertEqual(manifest['datasets']['ifeval']['max_new_tokens'], 1024)
            for name in entries:
                saved = run.read_rows(output/name/'results.jsonl')
                self.assertEqual([r['strength'] for r in saved], [0., 1.])
                self.assertTrue(all(r['prompt'] == 'target-'+name for r in saved))
                vector, _ = run.read_vector(Path(manifest['datasets'][name]['vector']))
                torch.testing.assert_close(vector, torch.full((3,), 2.))  # No unit normalization.

    def test_transfer_keeps_target_prompt_cohort_and_zero_only_reuse(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root = Path(d)
            entries = {name: self.fixture(root, name, name) for name in ['gsm8k', 'svamp']}
            # Make source vectors actually different so hashes cannot falsely identify own +1 as transfer.
            path = Path(entries['svamp']['run'])/'style-vectors.pt'
            state = torch.load(path, weights_only=True); state['pair_mean_sums']['angry'][31, 0] = 7
            torch.save(state, path)
            catalog = root/'catalog.json'; catalog.write_text(json.dumps(entries))
            output = root/'transfer'; output.mkdir()
            manifest = run.prepare(catalog, output, 'transfer')
            self.assertEqual(len(manifest['jobs']), 2)
            plan = json.loads((output/'gsm8k__from__svamp'/'run_config.json').read_text())
            self.assertEqual(plan['target']['prompt'], 'target-gsm8k')
            self.assertEqual(plan['source']['vector_info']['collection_prompt'], 'collection-svamp')
            self.assertEqual([r['question_id'] for r in plan['target']['records']], ['gsm8k_test'])
            self.assertEqual([r['strength'] for r in run.read_rows(output/'gsm8k__from__svamp'/'results.jsonl')], [0.])

    def test_changed_prompt_and_audio_cannot_reuse_old_answers(self):
        with tempfile.TemporaryDirectory() as d:
            entry = self.fixture(Path(d)); target = run.load_dataset('gsm8k', entry)
            config = json.loads(Path(target['config_file']).read_text()); record = target['records'][0]
            row = run.read_rows(Path(entry['run'])/'results.jsonl')[0]
            self.assertTrue(run.reusable(row, config, record, target))
            self.assertFalse(run.reusable(dict(row, prompt='different'), config, record, target))
            self.assertFalse(run.reusable(dict(row, audio_sha256='different'), config, record, target))

    def test_signed_strengths_hit_existing_hook_and_resume_without_repeating(self):
        class Block(torch.nn.Module):
            def forward(self, hidden): return (hidden,)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); entry = self.fixture(root); target = run.load_dataset('gsm8k', entry)
            plan = dict(target=target, source=target, strengths=[-.5, 0., .5])
            folder = root/'new'; folder.mkdir()
            layers = torch.nn.ModuleList([Block() for _ in range(32)])
            embedding = torch.nn.Embedding(2, 3)
            model = SimpleNamespace(llm_model=SimpleNamespace(get_input_embeddings=lambda: embedding,
                model=SimpleNamespace(transformer=SimpleNamespace(blocks=layers))))
            calls = []
            def generate(audio, sr, model, tokenizer, features, prompt, tokens, chunk):
                hidden = torch.zeros(1, tokens+3, 3)
                for layer in layers: hidden = layer(hidden)[0]
                calls.append((float(hidden[0, -1].mean()), float(hidden[0, 0].mean()), prompt, tokens))
                return '#### 10', {'mode': 'native'}
            original_to = torch.Tensor.to
            def cpu_to(tensor, *a, **kw):
                if kw.get('device') == 'cuda': kw['device'] = 'cpu'
                return original_to(tensor, *a, **kw)
            modules = {'soundfile': SimpleNamespace(read=lambda *a, **kw: (np.zeros(10), 16000)),
                       'extract_vectors': SimpleNamespace(gen_chunked=generate)}
            with patch.dict('sys.modules', modules), patch.object(torch.Tensor, 'to', cpu_to), redirect_stdout(io.StringIO()):
                run.generate_job(plan, folder, model, None, None)
                run.generate_job(plan, folder, model, None, None)
            self.assertEqual([c[0] for c in calls], [-1., 0., 1.])
            self.assertTrue(all(c[1:] == (0., 'target-gsm8k', 512) for c in calls))
            self.assertEqual(len(run.read_rows(folder/'results.jsonl')), 3)
            self.assertFalse(any(layer._forward_hooks for layer in layers))

    def test_numeric_and_bbh_use_upstream_task_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); entry = self.fixture(root, 'svamp', 'svamp'); target = run.load_dataset('svamp', entry)
            row = dict(target['records'][0], response='#### 10', strength=0., condition='neutral')
            scored, provenance = run.score_rows(dict(target=target), [row], root)
            self.assertTrue(scored[0]['metrics']['strict-match'])
            self.assertTrue(scored[0]['metrics']['flexible-numeric'])
            self.assertTrue(scored[0]['metrics']['author-tolerance'])
            self.assertEqual(provenance['task'], 'gsm8k')
            entry = self.fixture(root, 'bbh3', 'bbh3'); target = run.load_dataset('bbh3', entry)
            row = dict(target['records'][0], response='The answer is (A).', strength=0., condition='neutral')
            scored, provenance = run.score_rows(dict(target=target), [row], root/'bbh-score')
            self.assertTrue(scored[0]['metrics']['strict-match'])
            self.assertIn('logical_deduction_three_objects', provenance['task'])
            entry = self.fixture(root, 'bbh5', 'bbh5'); target = run.load_dataset('bbh5', entry)
            qid = target['records'][0]['question_id']
            target['bank'][qid].update(original_input='Which?\nOptions:\n(A) A\n(B) B\n(C) C\n(D) D\n(E) E', original_target='(E)')
            row = dict(target['records'][0], response='The answer is (E).', reference_answer='E', strength=0., condition='neutral')
            scored, provenance = run.score_rows(dict(target=target), [row], root/'bbh5-score')
            self.assertTrue(scored[0]['metrics']['strict-match'])
            self.assertIn('logical_deduction_five_objects', provenance['task'])

    def test_transfer_summary_has_one_nonzero_matrix_cell_and_preserves_raw_answers(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root = Path(d)
            target = run.load_dataset('gsm8k', self.fixture(root))
            source = run.load_dataset('svamp', self.fixture(root, 'svamp', 'svamp'))
            folder = root/'gsm8k__from__svamp'; folder.mkdir()
            plan = dict(source=source, target=target, strengths=[0., 1.])
            (folder/'run_config.json').write_text(json.dumps(plan))
            row = target['records'][0]
            rows = [dict(row, response='#### 10' if a == 0 else '#### 11', strength=a, condition=run.condition(a)) for a in (0., 1.)]
            path = folder/'results.jsonl'; path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            before = path.read_bytes()
            run.report_job(folder)
            manifest = dict(kind='transfer', datasets={'gsm8k': target, 'svamp': source},
                jobs=[dict(directory=folder.name)], note='exploratory')
            run.combined_report(root, manifest, rescore=False)
            with (root/'transfer-matrix.csv').open() as stream:
                cells = list(csv.DictReader(stream))
            self.assertEqual(len(cells), 1)
            self.assertEqual((cells[0]['source'], cells[0]['target'], float(cells[0]['strength'])), ('svamp', 'gsm8k', 1.))
            self.assertEqual(float(cells[0]['gain_pp']), -100.)
            self.assertEqual(path.read_bytes(), before)

    def test_missing_neutral_question_cannot_silently_shrink_frozen_bank(self):
        with tempfile.TemporaryDirectory() as d:
            entry = self.fixture(Path(d))
            path = Path(entry['run'])/'run_config.json'
            config = json.loads(path.read_text()); config['bank']['missing'] = dict(answer='10')
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'full frozen'):
                run.load_dataset('gsm8k', entry)

    def test_ifeval_routes_original_examples_to_official_checker_adapter(self):
        import run_spoken_ifeval as ifeval
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); entry = self.fixture(root, 'ifeval', 'ifeval'); target = run.load_dataset('ifeval', entry)
            row = dict(target['records'][0], response='Two words', strength=0., condition='neutral')
            def score(example, response, evaluator):
                self.assertEqual(example['key'], 1); self.assertEqual(response, 'Two words')
                return dict(strict={'prompt': True, 'instructions': [True]}, loose={'prompt': False, 'instructions': [False]})
            with patch.object(ifeval, 'setup_official'), patch.object(ifeval, 'official_scorer', return_value=object()), \
                 patch.object(ifeval, 'score', side_effect=score):
                scored, provenance = run.score_rows(dict(target=target), [row], root)
            self.assertEqual(scored[0]['metrics'], {'strict_prompt': True, 'loose_prompt': False})
            self.assertEqual(provenance['revision'], ifeval.REVISION)


if __name__ == '__main__': unittest.main()
