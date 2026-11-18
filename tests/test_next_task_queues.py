import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import phi4_task_steering as phi
import run_multiarith_second_voice as voice


class NextTaskTests(unittest.TestCase):
    def test_phi_preparation_has_disjoint_halves_and_no_model_then_explicit_plan(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'svamp';folder.mkdir()
            content=b'[]';(folder/'source.json').write_bytes(content)
            spec=dict(filename='source.json',sha=hashlib.sha256(content).hexdigest(),half=2,sources=[])
            questions=[dict(question_id=f'q{i}',question='Q',answer='2') for i in range(4)]
            records={(q['question_id'],s):dict(question_id=q['question_id'],voice='Jenny',style=s,answer='2',
                audio_file=f'audio-{i}-{s}',audio_sha256=f'{i}-{s}') for i,q in enumerate(questions) for s in ('neutral','angry')}
            with patch.dict(phi.TASKS,svamp=spec), patch.object(phi,'questions',return_value=(questions,'Prompt')), \
                 patch.object(phi,'prepare_audio') as audio, patch.object(phi.reasoning,'source_records',return_value=records), \
                 patch.object(phi.engine,'NativePhi') as model:
                data=phi.prepare_data('svamp',root)
                model.assert_not_called();audio.assert_called_once()
            self.assertEqual([p['question_id'] for p in data['collection']],['q0','q1'])
            self.assertEqual([p['question_id'] for p in data['test']],['q2','q3'])
            plan=phi.make_plan(data,folder/'data_config.json',30,'audio',-1.5)
            self.assertEqual(len(plan['recordings']),6)
            self.assertEqual({r['question_id'] for r in plan['recordings']},{'q2','q3'})
            self.assertEqual([r['strength'] for r in plan['recordings']], [0.,0.,-1.5]*2)
            self.assertEqual(plan['group'],'audio');self.assertEqual(plan['layer'],30)

    def test_selected_layer_capture_and_intervention_strength(self):
        class Block(torch.nn.Module):
            def forward(self,h):return h+1,'cache'
        layers=torch.nn.ModuleList([Block(),Block(),Block()]);mask=torch.tensor([True,False])
        with phi.native_hooks.hooks(layers,mask,collect=True,capture_layers=[1]) as (totals,counts,audio):
            for layer in layers:layer(torch.zeros(1,2,3))
        self.assertEqual(set(totals),{1});self.assertEqual(counts,{1:1})
        with phi.native_hooks.hooks(layers,mask,target=1,group='answer',direction=torch.ones(3),strength=-1.5):
            output=layers[1](torch.zeros(1,2,3))
        torch.testing.assert_close(output[0][0,-1],torch.full((3,),-.5))
        self.assertEqual(output[1],'cache');self.assertTrue(all(not b._forward_hooks for b in layers))

    def test_new_voice_preserves_original_290_test_ids_and_prompt(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);original=root/'original';original.mkdir()
            bank={f'q{i}':dict(question_id=f'q{i}',question='Q',answer='2') for i in range(291)}
            ids=list(bank)[:290]
            plan=dict(task='multiarith',decoder='fast_dllm',seed=42,max_new_tokens=512,prompt='Same prompt',bank=bank,
                recordings=[dict(question_id=q,condition='neutral') for q in ids])
            (original/'run_config.json').write_text(json.dumps(plan))
            records={(q,s):dict(question_id=q,voice='Guy',style=s,answer='2',audio_file='a',audio_sha256=q+s)
                     for q in ids for s in ('neutral','angry')}
            args=SimpleNamespace(original=original,source=root/'audio',output=root/'run',voice='en-US-GuyNeural')
            with patch.object(voice.subprocess,'run') as tts,patch.object(voice.reasoning,'source_records',return_value=records):
                prepared=voice.prepare(args)
            self.assertEqual(prepared['prompt'],'Same prompt');self.assertEqual(len(prepared['recordings']),580)
            self.assertNotIn('collection',prepared);self.assertNotIn('vector',prepared)
            self.assertNotIn('q290',prepared['bank'])
            command=tts.call_args.args[0]
            self.assertEqual(command[command.index('--azure-voice')+1],'en-US-GuyNeural')
            self.assertEqual(command[command.index('--styles')+1:command.index('--output')],['neutral','angry'])


if __name__=='__main__':unittest.main()
