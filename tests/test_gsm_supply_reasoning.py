import io
import json
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import run_gsm_supply_reasoning as run


class GsmSupplyTests(unittest.TestCase):
    dataset = 'gsm8k'
    pool_size = 1319

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.parent = Path(temp.name)/'parent'
        self.output = Path(temp.name)/'out'
        bank = {f'q{i}':dict(question='One plus one?',answer='2') for i in range(self.pool_size)}
        self.ids = random.Random(42).sample(sorted(bank),200)
        records = [dict(question_id=q,voice='en-US-JennyNeural',style=s,condition=s,
                        audio_sha256=q+s,audio_file='fixture.wav',reference_answer='2')
                   for q in bank for s in run.STYLES]
        self.original = dict(protocol=run.night.PROTOCOL,model='diffa',task=self.dataset,bank=bank,
            prompt=run.night.vanilla.PROMPT,seed=42,max_new_tokens=512,recipe=run.night.vanilla.RECIPE,records=records)
        config = self.parent/'run_config.json'
        run.winners.freeze(config,self.original)
        checksum = run.winners.digest(config)
        self.rows = [dict(r,model='diffa',prompt=self.original['prompt'],run_config_sha256=checksum,
            strength=0.,interventions=[],response='#### 1',response_tokens=3,**run.night.vanilla.RECIPE)
            for r in records if r['question_id'] in self.ids]
        self.write_rows()

    def write_rows(self):
        (self.parent/'deliveries.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))

    def test_sample_same_clips_prompt_and_resume(self):
        plan = run.prepare(self.parent,self.output,self.dataset)
        self.assertEqual(plan['task'], self.dataset)
        self.assertEqual(plan['question_order'],self.ids)
        self.assertEqual(len(set(self.ids)),200)
        self.assertEqual(len(plan['records']),400)
        self.assertEqual(run.PROMPT.replace(' Supply reasoning before the final answer.',''),plan['bare_prompt'])
        self.assertEqual([r['audio_sha256'] for r in plan['records']],[r['audio_sha256'] for r in plan['bare_answers']])
        calls = []
        class Backend:
            def generate(this,record,scoped,capture):
                self.assertEqual(scoped['prompt'],run.PROMPT)
                self.assertFalse(capture)
                calls.append((record['question_id'],record['style']))
                return dict(response='#### 2' if record['style']=='angry' else '#### 1',
                            response_tokens=10 if record['style']=='angry' else 3,audio_encoding={})
        with redirect_stdout(io.StringIO()):
            run.evaluation.generate_dataset(self.output,plan,plan['records'][:1],Backend())
            run.evaluation.generate_dataset(self.output,plan,plan['records'][:3],Backend())
            run.report(self.output,plan)
        self.assertEqual(len(calls),3)
        summary = json.loads((self.output/'summary.json').read_text())
        self.assertEqual(summary['matched'],1)
        self.assertEqual(summary['lengths']['angry-reasoning']['mean_tokens'],10)
        for contrast in summary['interactions']:
            self.assertEqual(contrast['interaction_pp'],0 if contrast['metric']=='format-compliant' else 100)
        self.assertEqual(set(c['metric'] for c in summary['comparisons']),
                         {'strict-match','flexible-extract','flexible-numeric','format-compliant'})

    def test_no_resampling_on_missing_or_mismatched_bare_answer(self):
        self.rows.pop()
        self.write_rows()
        with self.assertRaisesRegex(ValueError,'Missing saved bare'):
            run.prepare(self.parent,self.output,self.dataset)
        self.rows[0]['audio_sha256']='wrong'
        self.write_rows()
        with self.assertRaisesRegex(ValueError,'cache differs'):
            run.prepare(self.parent,self.output,self.dataset)


class MultiArithSupplyTests(GsmSupplyTests):
    dataset = 'multiarith'
    pool_size = 580

    def test_wrong_parent_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'GSM8K'):
            run.prepare(self.parent, self.output, 'gsm8k')

    def test_supply_only_from_audio_without_bare_responses(self):
        from unittest.mock import Mock
        registry = Mock()
        bank = self.original['bank']
        registry.root = self.parent
        registry.neutral_ids.return_value = list(bank)
        registry.options = {}
        audio = self.parent/'clip.wav'
        audio.write_bytes(b'fixture')
        for row in self.original['records']:
            item = dict(record=dict(row, audio_file=str(audio), audio_sha256=run.winners.digest(audio)),
                        **bank[row['question_id']])
            registry.options['multiarith',row['question_id'],row['style']] = [item]
        registry.choose.side_effect = lambda *key: registry.options[key][0]
        with patch('phi4_more_bare_eval.Registry', return_value=registry):
            plan = run.prepare_supply_only(self.output, audio_root=self.parent)
        self.assertEqual(plan['question_order'], self.ids)
        self.assertNotIn('bare_answers', plan)
        backend = Mock()
        backend.generate.return_value = dict(response='#### 2', response_tokens=3, audio_encoding={})
        with redirect_stdout(io.StringIO()):
            run.evaluation.generate_dataset(self.output, plan, plan['records'][:3], backend)
            run.evaluation.generate_dataset(self.output, plan, plan['records'][:3], backend)
            run.report(self.output, plan)
        self.assertEqual(backend.generate.call_count, 3)
        summary = json.loads((self.output/'summary.json').read_text())
        self.assertEqual(summary['matched'], 1)
        self.assertEqual(summary['interactions'], [])
        self.assertEqual(set(summary['lengths']), {'neutral-reasoning', 'angry-reasoning'})
        self.assertEqual(len(summary['comparisons']), 4)
        self.assertTrue(all(c['prompt'] == 'supply' and c['baseline_correct'] == c['steered_correct'] == 1
                            for c in summary['comparisons']))
