from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np
import torch
import diffa_layer_probe as probe


class LayerProbeTests(unittest.TestCase):
    def fixture(self, root, regime='random'):
        source = root/'outputs/diffa-gsm-vanilla/angry'/regime
        source.mkdir(parents=True)
        records = []
        for q in ('q0', 'q1'):
            audio = root/(q+'.wav')
            with wave.open(str(audio), 'wb') as stream:
                stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(16000)
                stream.writeframes(b'\0\0'*160)
            records.append(dict(question_id=q, voice='Jenny', style='neutral', condition='neutral',
                audio_file=str(audio), audio_sha256=probe.vanilla.task.winners.digest(audio), reference_answer='2'))
        vector = source/'style-vectors.pt'
        sums = torch.zeros(32, 4, 3); sums[28, 3] = 4.; sums[31, 3] = 10.
        torch.save(dict(mode='fixed', method='mean', groups=['acoustic', 'semantic', 'combined', 'answer'],
            pair_mean_sums={'angry': sums}, pair_counts={'angry': torch.full((32,), 2)},
            extraction_question_ids=['fit0', 'fit1'], prompt=probe.vanilla.PROMPT, max_new_tokens=512), vector)
        target = dict(name='gsm8k', task='gsm8k', records=records, prompt=probe.vanilla.PROMPT,
            **probe.vanilla.RECIPE, bank={r['question_id']: dict(question='Question', answer='2') for r in records},
            vector=str(vector), vector_sha256=probe.vanilla.task.winners.digest(vector), emotion='angry', primary_metric='flexible-numeric')
        job = dict(source=target, target=target, emotion='angry', regime=regime, layer=31, group='answer', strengths=[0., 1.])
        original = source/'strength-1-unfitted'; original.mkdir()
        (original/'run_config.json').write_text(json.dumps(job))
        delivery = source.parent.parent/'delivery'; delivery.mkdir()
        rows = [dict(r, response='#### 2', response_tokens=3, **probe.vanilla.RECIPE, prompt=probe.vanilla.PROMPT,
                     strength=0., interventions=[]) for r in records]
        (delivery/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        revisions = {m: str(i)*40 for i, m in enumerate(probe.MODELS, 1)}
        destination = root/f'outputs/diffa-layer-probes/gsm8k-angry-{regime}-L28-answer-a1'
        with patch.object(probe, 'model_revisions', return_value=revisions), patch.object(probe, 'version', return_value='test'), redirect_stdout(io.StringIO()):
            archive = probe.package(source, destination, root)
        return archive, destination

    def test_pack_restore_keeps_cohort_and_continues_appended_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, folder = self.fixture(root)
            probe.unpack(archive, root)
            job = json.loads((folder/'run_config.json').read_text())
            self.assertEqual(job['layer'], 28)
            self.assertEqual(len(probe.validate(job, folder, check_audio=True)), 2)
            value28, info28 = probe.vanilla.task.winners.read_vector(job['source']['vector'], layer=28)
            value31, info31 = probe.vanilla.task.winners.read_vector(job['source']['vector'])
            torch.testing.assert_close(value28, torch.full((3,), 2.))
            torch.testing.assert_close(value31, torch.full((3,), 5.))
            self.assertEqual(info28, info31)
            record = job['target']['records'][0]
            row = dict(record, **probe.vanilla.RECIPE, prompt=probe.vanilla.PROMPT, response='#### 2',
                       condition='L28-answer-a1', strength=1., vector_sha256=job['source']['vector_sha256'],
                       interventions=[dict(layer=28, group='answer', strength=1.)])
            with (folder/'results.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            probe.unpack(archive, root)
            self.assertEqual(len(probe.validate(job, folder)), 3)
            with self.assertRaises(ValueError):
                probe.unpack(archive, root/'another-root')

    def test_generation_uses_layer28_values_and_hooks_and_resumes(self):
        class Block(torch.nn.Module):
            def forward(self, hidden):
                return (hidden,)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, folder = self.fixture(root)
            probe.unpack(archive, root)
            job = json.loads((folder/'run_config.json').read_text())
            blocks = torch.nn.ModuleList([Block() for _ in range(32)])
            model = SimpleNamespace(llm_model=SimpleNamespace(get_input_embeddings=lambda: torch.nn.Embedding(1, 3),
                model=SimpleNamespace(transformer=SimpleNamespace(blocks=blocks))))
            observed = []
            def generate(*args):
                hidden = torch.zeros(1, 515, 3)
                for block in blocks:
                    hidden = block(hidden)[0]
                observed.append((float(hidden[0, -1].mean()), float(hidden[0, 0].mean())))
                return '#### 2', {}
            original_to = torch.Tensor.to
            def cpu_to(tensor, *args, **kw):
                if kw.get('device') == 'cuda':
                    kw['device'] = 'cpu'
                return original_to(tensor, *args, **kw)
            modules = {'extract_vectors': SimpleNamespace(gen_chunked=generate),
                       'soundfile': SimpleNamespace(read=lambda *a, **kw: (np.zeros(10), 16000))}
            tokenizer = SimpleNamespace(encode=lambda *a, **kw: [1, 2, 3])
            with patch.dict('sys.modules', modules), patch.object(torch.Tensor, 'to', cpu_to), redirect_stdout(io.StringIO()):
                probe.vanilla.task.winners.generate_job(job, folder, model, tokenizer, None)
                probe.vanilla.task.winners.generate_job(job, folder, model, tokenizer, None)
            self.assertEqual(observed, [(2., 0.), (2., 0.)])
            self.assertEqual(len(probe.validate(job, folder)), 4)
            self.assertTrue(all(row['response_tokens'] == 3 for row in probe.validate(job, folder).values()))
            self.assertTrue(all(not block._forward_hooks for block in blocks))

    def test_audio_relocation_uses_exact_zip_hash_without_overwriting_old_phi_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, folder = self.fixture(root)
            probe.unpack(archive, root)
            config_file = folder/'run_config.json'
            original_config = config_file.read_bytes()
            job = json.loads(original_config)
            audio_root = root/'outputs/jenny-zip-audio'
            audio_root.mkdir(parents=True)
            entries = []
            for record in job['target']['records']:
                original = Path(record['audio_file'])
                copied = audio_root/original.name
                copied.write_bytes(original.read_bytes())
                original.write_bytes(b'older Phi recording')
                entries.append(dict(member=copied.name, audio_sha256=record['audio_sha256']))
            (audio_root/'JENNY_AUDIO_MANIFEST.json').write_text(json.dumps(dict(files=entries)))
            paths, mappings = probe.resolve_audio(job, [audio_root, root])
            relocated = probe.with_audio_paths(job, paths)
            self.assertEqual(len(probe.validate(relocated, folder, check_audio=True)), 2)
            self.assertEqual(len(mappings), 2)
            self.assertTrue(all(Path(row['runtime_audio_file']).parent == audio_root for row in mappings))
            self.assertEqual(config_file.read_bytes(), original_config)
            self.assertTrue(all(Path(r['audio_file']).read_bytes() == b'older Phi recording' for r in job['target']['records']))
            with self.assertRaisesRegex(ValueError, 'Exact audio bytes not found'):
                probe.resolve_audio(job, [root/'empty-root'])

    def test_agreed_sweep_uses_identical_seeded_subset_and_shared_vector(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, regime='agreed')
            source = root/'outputs/diffa-gsm-vanilla/angry/agreed'
            destination = root/'outputs/diffa-layer-probes/angry-agreed-L25-30-n1'
            revisions = {m: str(i)*40 for i, m in enumerate(probe.MODELS, 1)}
            with patch.object(probe, 'model_revisions', return_value=revisions), patch.object(probe, 'version', return_value='test'), redirect_stdout(io.StringIO()):
                archive = probe.package(source, destination, root, layer=25, layers=list(range(25, 31)), questions=1)
            folder = probe.unpack(archive, root)
            job = json.loads((folder/'run_config.json').read_text())
            receipt = json.loads((folder/'bundle-receipt.json').read_text())
            cases = probe.probe_jobs(folder, job, receipt)
            self.assertEqual([j['layer'] for _, j in cases], list(range(25, 31)))
            self.assertEqual({j['regime'] for _, j in cases}, {'agreed'})
            question_lists = [[r['question_id'] for r in j['target']['records']] for _, j in cases]
            self.assertTrue(all(ids == question_lists[0] and len(ids) == 1 for ids in question_lists))
            self.assertEqual(len({j['source']['vector'] for _, j in cases}), 1)
            for child, selected in cases:
                self.assertEqual(len(probe.validate(selected, child)), 1)
                record = selected['target']['records'][0]
                row = dict(record, **probe.vanilla.RECIPE, prompt=probe.vanilla.PROMPT, response='#### 2',
                    condition=f'L{selected["layer"]}-answer-a1', strength=1., vector_sha256=selected['source']['vector_sha256'],
                    interventions=[dict(layer=selected['layer'], group='answer', strength=1.)])
                with (child/'results.jsonl').open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
            probe.unpack(archive, root)
            for child, selected in probe.probe_jobs(folder, job, receipt):
                self.assertEqual(len(probe.validate(selected, child)), 2)
            with self.assertRaisesRegex(ValueError, 'test questions'):
                probe.package(source, root/'impossible', root, questions=3)

    def test_pack_agreed_before_l31_agreed_steering_has_started(self):
        from tests.test_diffa_gsm_vanilla import VanillaTests
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root/'outputs/diffa-gsm-vanilla'
            helper = VanillaTests()
            manifest, plans = helper.prepare(run)
            scoped = probe.vanilla.prepare_deliveries(helper.registry(), manifest, plans, run)
            tokenizer = SimpleNamespace(encode=lambda *a, **kw: [1, 2])
            with patch.object(probe.vanilla.task, 'generate', return_value=('#### 2', {})):
                probe.vanilla.deliver(scoped, ['angry'], run/'delivery', None, tokenizer, None)
            source = run/'angry/agreed'
            train = plans['angry']['agreed']['collection_ids']
            torch.save(dict(mode='fixed', method='mean', groups=['acoustic', 'semantic', 'combined', 'answer'],
                pair_mean_sums={'angry': torch.ones(32, 4, 3)}, pair_counts={'angry': torch.full((32,), len(train))},
                extraction_question_ids=train, prompt=probe.vanilla.PROMPT, max_new_tokens=512), source/'style-vectors.pt')
            self.assertFalse((source/'strength-1-unfitted/run_config.json').exists())
            revisions = {m: str(i)*40 for i, m in enumerate(probe.MODELS, 1)}
            with patch.object(probe, 'model_revisions', return_value=revisions), patch.object(probe, 'version', return_value='test'), redirect_stdout(io.StringIO()):
                archive = probe.package(source, root/'outputs/probe-early-agreed', root, layer=25, layers=list(range(25, 31)), questions=2)
            folder = probe.unpack(archive, root)
            job = json.loads((folder/'run_config.json').read_text())
            self.assertEqual(job['regime'], 'agreed')
            self.assertEqual(len(job['target']['records']), 2)
            self.assertFalse(set(train) & {r['question_id'] for r in job['target']['records']})

    def test_natural_benefit_scores_only_exact_probe_questions_with_same_baseline(self):
        from tests.test_diffa_gsm_vanilla import VanillaTests
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root/'original'
            helper = VanillaTests()
            manifest, plans = helper.prepare(run)
            scoped = probe.vanilla.prepare_deliveries(helper.registry(), manifest, plans, run)
            tokenizer = SimpleNamespace(encode=lambda *a, **kw: [1, 2])
            def generate(record, *args):
                return ('#### 3' if record['style'] == 'angry' and record['question_id'] == 'q00' else '#### 2'), {}
            with patch.object(probe.vanilla.task, 'generate', side_effect=generate):
                probe.vanilla.deliver(scoped, ['angry'], run/'delivery', None, tokenizer, None)
            target = dict(scoped['angry']['agreed']['target'])
            wanted = {'q00', 'q03'}
            target['records'] = [r for r in target['records'] if r['question_id'] in wanted]
            target['bank'] = {q: target['bank'][q] for q in wanted}
            job = dict(target=target, emotion='angry', split_sha256=probe.vanilla.fingerprint(manifest))
            baselines = [r for r in probe.read_rows(run/'delivery/results.jsonl')
                         if r['question_id'] in wanted and r['condition'] == 'neutral']
            output = root/'report'
            def score(plan, rows, folder):
                return [dict(r, metrics={'flexible-numeric': r['response'] == '#### 2'}) for r in rows], {}
            with patch.object(probe.vanilla.task.winners, 'score_rows', side_effect=score):
                probe.report_natural_benefit(job, baselines, run, output)
            summary = json.loads((output/'summary.json').read_text())
            self.assertEqual(summary['matched_questions'], 2)
            self.assertEqual(summary['paired'][0]['baseline_correct'], 2)
            self.assertEqual(summary['paired'][0]['steered_correct'], 1)
            # q00 is not angry-recognized, but must stay in this exact200-style cohort.
            self.assertIn('q00', json.loads((output/'selection.json').read_text())['question_ids'])
            baselines[0]['response'] = 'different neutral'
            with self.assertRaisesRegex(ValueError, 'exact baseline'):
                probe.report_natural_benefit(job, baselines, run, output)

    def test_fork_layers_preserves_exact_questions_vector_and_baseline_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, _ = self.fixture(root, regime='agreed')
            original_bytes = reference.read_bytes()
            forked = []
            with redirect_stdout(io.StringIO()):
                for layers in (list(range(15, 26)), list(range(5, 16))):
                    archive = probe.fork_layers(reference, root, layers)
                    self.assertEqual(archive, probe.fork_layers(reference, root, layers))
                    folder = probe.unpack(archive, root)
                    job = json.loads((folder/'run_config.json').read_text())
                    receipt = json.loads((folder/'bundle-receipt.json').read_text())
                    cases = probe.probe_jobs(folder, job, receipt)
                    self.assertEqual([j['layer'] for _, j in cases], layers)
                    self.assertEqual(job['emotion'], 'angry')
                    self.assertEqual(job['regime'], 'agreed')
                    self.assertEqual(job['strengths'], [0., 1.])
                    self.assertEqual(len(cases), 11)
                    self.assertTrue(all(len(j['target']['records']) == 2 for _, j in cases))
                    self.assertTrue(all(len(probe.validate(j, child)) == 2 for child, j in cases))
                    self.assertEqual(probe.fork_layers(archive, root, layers), archive)
                    forked.append((folder, job))
            left, right = forked
            self.assertEqual(left[1]['target']['records'], right[1]['target']['records'])
            self.assertEqual(left[1]['source']['vector_sha256'], right[1]['source']['vector_sha256'])
            self.assertEqual((left[0]/'results.jsonl').read_bytes(), (right[0]/'results.jsonl').read_bytes())
            self.assertEqual(reference.read_bytes(), original_bytes)


if __name__ == '__main__':
    unittest.main()
