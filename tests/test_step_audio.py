"""Step delivery identity, full audio windows, suffix decoding and queued execution."""
from contextlib import redirect_stdout
import io,json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock,patch
import torch

import step_audio_adapter as adapter
import evaluate_step_gsm as run
from kimi_bbh_layer_diagnostic import digest


class StepTests(unittest.TestCase):
    def test_full_audio_uses_native25second_windows_with_same_prompt(self):
        calls=[];parts=[]
        class Tokenizer:
            def __call__(self,text,**kwargs):
                parts.append(text);return {'input_ids':torch.tensor([[1]])}
        def mel(audio,**kwargs):
            calls.append(len(audio));return torch.zeros(128,10)
        features=SimpleNamespace(load_audio=lambda path:torch.zeros(51*16000),log_mel_spectrogram=mel,compute_token_num=lambda n:2)
        inputs=adapter.native_inputs(Tokenizer(),features,'audio.wav','Task prompt','cpu')
        self.assertEqual(calls,[25*16000,25*16000,16000])
        self.assertEqual(parts[2],'Task prompt')
        self.assertEqual(parts.count('<audio_start><audio_patch><audio_patch><audio_end>'),3)
        self.assertEqual(parts[-1],'<|BOT|>assistant\n')
        self.assertNotIn('<tts_start>',''.join(parts))
        self.assertEqual(inputs['wavs'].shape,(3,128,10))
        self.assertEqual(inputs['wav_lens'].tolist(),[8,8,8])

    def test_eos_removed_only_if_present_not_blind_last_token_drop(self):
        tokenizer=SimpleNamespace(decode=lambda ids:','.join(map(str,ids)))
        self.assertEqual(adapter.decode_suffix(tokenizer,torch.tensor([12,34,151665]),151665),('12,34',3))
        self.assertEqual(adapter.decode_suffix(tokenizer,torch.tensor([12,34]),151665),('12,34',2))
        self.assertEqual(adapter.decode_suffix(tokenizer,torch.tensor([12,151700,34,151665]),151665),('12,34',4))

    def test_generator_uses_native_audio_and_greedy512_without_gold(self):
        class Model:
            device='cpu'
            config=SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=16384))
            def eval(self):return self
            def cuda(self):return self
            def generate(self,**kwargs):
                self.kwargs=kwargs;return torch.tensor([[1,2,7,8,151665]])
        model=Model()
        tokenizer=SimpleNamespace(pad_token_id=151643,convert_tokens_to_ids=lambda s:151665,decode=lambda ids:'answer '+str(ids))
        auto=lambda obj:SimpleNamespace(from_pretrained=lambda *a,**k:obj)
        stub=SimpleNamespace(AutoModelForCausalLM=auto(model),AutoTokenizer=auto(tokenizer),GenerationConfig=SimpleNamespace)
        with patch.dict('sys.modules',{'transformers':stub}),patch.object(adapter,'native_inputs',return_value={'input_ids':torch.tensor([[1,2]])}) as inputs:
            response,tokens=adapter.make_generator('/pinned')({'audio_file':'voice.wav','reference_answer':'SECRET'},'EXACT PROMPT')
        self.assertEqual((response,tokens),('answer [7, 8]',3))
        self.assertEqual(inputs.call_args.args[2:4],('voice.wav','EXACT PROMPT'))
        self.assertFalse(model.kwargs['generation_config'].do_sample)
        self.assertEqual(model.kwargs['generation_config'].max_new_tokens,512)
        self.assertNotIn('reference_answer',model.kwargs)

    def fixture(self,root):
        audio=root/'audio.wav';audio.write_bytes(b'fixed audio identity')
        bank={f'q{i}':dict(question_id=f'q{i}',question='Question',answer='2') for i in range(10)}
        records=[dict(question_id=qid,voice='Jenny',style=s,condition=s,answer='2',audio_file=str(audio),
            audio_sha256=digest(audio),seconds=45) for qid in bank for s in ('neutral','angry')]
        records += [dict(records[0],condition='neutral-steered')]
        source=root/'source.json';source.write_text(json.dumps(dict(task='gsm8k',bank=bank,recordings=records,prompt='Reason, then #### <number>.')))
        return SimpleNamespace(source=source,output=root/'out',num_questions=4)

    def test_paired_fixed_subset_no_steering_no_source_mutation_and_freeze(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            args=self.fixture(Path(tmp));before=args.source.read_bytes();plan=run.prepare(args)
            self.assertEqual(len(plan['bank']),4);self.assertEqual(len(plan['recordings']),8)
            self.assertEqual(plan['prompt'],'Reason, then #### <number>.')
            self.assertEqual(plan['interventions'],[])
            for qid in plan['bank']:
                self.assertEqual({r['style'] for r in plan['recordings'] if r['question_id']==qid},{'neutral','angry'})
            self.assertEqual(args.source.read_bytes(),before);self.assertEqual(run.prepare(args),plan)
            args.num_questions=5
            with self.assertRaises(ValueError):run.prepare(args)

    def test_changed_audio_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp));(Path(tmp)/'audio.wav').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'audio changed'):run.prepare(args)

    def test_missing_pair_is_not_silently_filtered(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp));args.num_questions=0
            source=json.loads(args.source.read_text());source['recordings']=[r for r in source['recordings'] if not (r['question_id']=='q1' and r['condition']=='angry')]
            args.source.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError,'Missing paired'):run.prepare(args)

    def test_gpu_model_load_waits_until_gpu_idle(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            args=self.fixture(Path(tmp));plan=run.prepare(args);events=[]
            with patch.object(run.candidates,'pending',return_value=[1]),patch.object(run.candidates,'run',side_effect=lambda *a:events.append('load')) as generate,\
                 patch.object(run.Path,'open',return_value=(Path(tmp)/'lock').open('a')),\
                 patch.object(run.subprocess,'check_output',side_effect=['123\n','']),\
                 patch.object(run.time,'sleep',side_effect=lambda n:events.append('wait')):
                # Separate handles for own/shared locks, as in the real filesystem.
                with patch.object(run.fcntl,'flock'):
                    run.evaluate(args,plan)
            self.assertEqual(events,['wait','load']);generate.assert_called_once()


class CalculationPromptTests(unittest.TestCase):
    fixture = StepTests.fixture
    def test_calculations_prompt_keeps_baseline_ids_and_cannot_overwrite_original(self):
        from step_calculations20 import PROMPT
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp);args=self.fixture(root);original=run.prepare(args)
            args.source=args.output/'run_config.json';args.output=root/'calculations';args.prompt='calculations'
            updated=run.prepare(args)
            self.assertEqual(list(updated['bank']),list(original['bank']))
            self.assertEqual(updated['recordings'],original['recordings'])
            self.assertEqual(updated['prompt'],PROMPT)
            args.output=root/'out'
            with self.assertRaises(ValueError):run.prepare(args)

    def test_reuse_maps_audio_only_and_rejects_changed_prompt(self):
        from step_calculations20 import PROMPT
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp);args=self.fixture(root);args.prompt='calculations';plan=run.prepare(args)
            diagnostic=root/'diagnostic';diagnostic.mkdir()
            config=dict(protocol='step-calculations-same20-v1',prompt=PROMPT,system_prompt=plan['system_prompt'],recipe=plan['recipe'],seed=42,max_new_tokens=512)
            (diagnostic/'run_config.json').write_text(json.dumps(config))
            records=[]
            for task in plan['recordings'][:2]:
                records.append(dict(task,condition=task['style']+'-calculations',prompt=PROMPT,seed=42,max_new_tokens=512,
                    model=plan['recipe']['model'],revision=plan['recipe']['revision'],interventions=[],response='#### 2',response_tokens=6))
            records.append(dict(question_id='text-only',condition='text-calculations',response='#### 2'))
            (diagnostic/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            self.assertEqual(run.reuse_diagnostic(plan,args.output,diagnostic),2)
            self.assertEqual(run.reuse_diagnostic(plan,args.output,diagnostic),0)
            saved=run.read_rows(args.output/'results.jsonl')
            self.assertEqual({r['condition'] for r in saved},{'neutral','angry'})
            self.assertEqual(len(run.candidates.pending([(args.output,plan)])),6)
            config['prompt']='different'
            (diagnostic/'run_config.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'mixed-prompt'):run.reuse_diagnostic(plan,args.output,diagnostic)


if __name__=='__main__':unittest.main()
