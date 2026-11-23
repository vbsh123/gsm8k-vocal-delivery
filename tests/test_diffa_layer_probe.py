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
            with patch.dict('sys.modules', modules), patch.object(torch.Tensor, 'to', cpu_to), redirect_stdout(io.StringIO()):
                probe.vanilla.task.winners.generate_job(job, folder, model, None, None)
                probe.vanilla.task.winners.generate_job(job, folder, model, None, None)
            self.assertEqual(observed, [(2., 0.), (2., 0.)])
            self.assertEqual(len(probe.validate(job, folder)), 4)
            self.assertTrue(all(not block._forward_hooks for block in blocks))

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


if __name__ == '__main__':
    unittest.main()
