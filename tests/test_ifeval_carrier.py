import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock
import xml.etree.ElementTree as ET

from cloud_tts_smoke import azure_ssml
import ifeval_carrier as carrier


class WrittenIFEvalTest(TestCase):
    def test_words_and_rates_match_between_styles(self):
        for rate in carrier.RATES:
            for style in carrier.STYLES:
                root = ET.fromstring(azure_ssml(carrier.PHRASE, style, 'en-US-JennyNeural', 1.5, rate))
                self.assertEqual(''.join(root.itertext()), carrier.PHRASE)
                prosody = root.find('.//{*}prosody')
                if rate:
                    self.assertEqual(prosody.get('rate'), f'{rate:+d}%')
                else:
                    self.assertIsNone(prosody)
                self.assertEqual(root.find('.//{*}express-as') is not None, style=='angry')

    def test_matched_prompts_and_resume(self):
        clips = {(rate,s):dict(voice='Jenny', audio_file=f'{rate}/{s}', audio_sha256=f'{rate}/{s}')
                 for rate in carrier.RATES for s in carrier.STYLES}
        bank = {f'ifeval_{i}':dict(prompt=f'Exact prompt {i}\n', instruction_id_list=['a']*(1+i%2))
                for i in range(24)}
        plan = carrier.build_plan(bank, clips)
        self.assertEqual(plan, carrier.build_plan(bank, clips))
        self.assertEqual(plan['recipe']['max_new_tokens'], 1024)
        records = plan['records']
        for neutral, angry in zip(records[::2], records[1::2]):
            self.assertEqual((neutral['style'], angry['style']), ('neutral', 'angry'))
            self.assertEqual(neutral['question_id'], angry['question_id'])
            self.assertEqual(neutral['prompt'], bank[neutral['question_id']]['prompt'])
            self.assertEqual(angry['prompt'], neutral['prompt'])
            self.assertEqual(neutral['carrier_rate'], angry['carrier_rate'])
            self.assertEqual(neutral['spoken_prompt'], carrier.PHRASE)
        class Backend:
            tokenizer = object()
            def generate(self, record, task, capture):
                assert task['prompt'] == bank[record['question_id']]['prompt']
                return dict(response=record['style'], response_tokens=2)
        with TemporaryDirectory() as tmp, mock.patch.object(carrier.pilot.evaluation, 'context_guard'):
            folder = Path(tmp)
            plan['records'] = records[:3]
            carrier.winners.freeze(folder/'run_config.json', plan)
            for _ in range(2):
                carrier.pilot.generate(folder, plan, Backend(), None)
            def score(_plan, rows, _folder):
                result = []
                for row in rows:
                    ok = row['response']=='angry'
                    bits = [ok]*len(bank[row['question_id']]['instruction_id_list'])
                    result.append(dict(row, metrics={m+'_prompt':ok for m in ('strict','loose')},
                                       official_scores={m:{'instructions':bits} for m in ('strict','loose')}))
                return result, {}
            with mock.patch.object(carrier.winners, 'score_rows', side_effect=score):
                carrier.report(folder, plan, folder/'missing-baseline')
            summary = json.loads((folder/'summary.json').read_text())
            self.assertEqual(summary['matched'], 1)
            self.assertEqual(summary['comparisons']['all']['strict_prompt']['wins'], 1)
            self.assertEqual(len(carrier.pilot.saved(folder, plan)), 3)
