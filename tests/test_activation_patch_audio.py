from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import wave

from activation_patch_audio import capture_prefix,patch_forward,select_pairs,ANSWER_TOKENS


class PatchSelectionTests(unittest.TestCase):
    def test_pairing_uses_question_voice_and_both_duration_limits_without_labels(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);tasks=[]
            for q,style,seconds in [('q1','neutral',1),('q1','angry',1),('q2','neutral',1),('q2','angry',31),('q3','neutral',1)]:
                name=f'{q}-{style}.wav'
                with wave.open(str(root/name),'wb') as f:
                    f.setnchannels(1);f.setsampwidth(2);f.setframerate(10);f.writeframes(b'\0\0'*seconds*10)
                tasks.append(dict(provider='azure',question_id=q,voice='Jenny',style=style,audio_path=name,answer='7'))
            (root/'plan.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in tasks))
            pairs=select_pairs([root],0)
            self.assertEqual([p['question_id'] for p in pairs],['q1'])
            self.assertEqual(select_pairs([root],1),pairs)
            with self.assertRaises(ValueError):select_pairs([root],2)


@unittest.skipUnless(importlib.util.find_spec('torch'),'Needs CPU PyTorch')
class MatchedStatePatchTests(unittest.TestCase):
    def model(self,torch):
        class Block(torch.nn.Module):
            def forward(self,x):return (x+x.mean(dim=1,keepdim=True),None)
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.blocks=torch.nn.ModuleList([Block(),Block()]);self.seen=[]
            def forward(self,inputs_embeds):
                self.seen.append(inputs_embeds.detach().clone());x=inputs_embeds
                for block in self.blocks:x=block(x)[0]
                return x
        return Model()

    def test_donor_gets_exact_current_answer_state_and_last_layer_patch_matches_angry(self):
        import torch
        model=self.model(torch);answer=torch.arange(ANSWER_TOKENS,dtype=torch.float32).reshape(1,-1,1)
        neutral=torch.cat([torch.zeros(1,3,1),answer],1);donor_prefix=torch.ones(1,5,1)*7
        expected=model(inputs_embeds=torch.cat([donor_prefix,answer],1))[:,-ANSWER_TOKENS:]
        model.seen.clear();bounds=dict(start=0,acoustic_end=2)
        with patch_forward(model,model.blocks[-1],donor_prefix,bounds,bounds,'answer',torch) as state:
            output=model(inputs_embeds=neutral)
            changed=neutral.clone();changed[:,-1]=999
            model(inputs_embeds=changed)
        torch.testing.assert_close(output[:,-ANSWER_TOKENS:],expected)
        torch.testing.assert_close(model.seen[0][:,:5],donor_prefix)
        torch.testing.assert_close(model.seen[0][:,-ANSWER_TOKENS:],answer)
        torch.testing.assert_close(model.seen[2][:,-ANSWER_TOKENS:],changed[:,-ANSWER_TOKENS:])
        self.assertEqual((state['donor_forwards'],state['recipient_forwards'],state['patched_forwards']),(2,2,2))
        self.assertFalse(model._forward_pre_hooks)
        self.assertFalse(model.blocks[-1]._forward_hooks)

    def test_self_patch_is_identity_and_acoustic_tokens_are_only_replaced_group(self):
        import torch
        model=self.model(torch);x=torch.arange(ANSWER_TOKENS+3,dtype=torch.float32).reshape(1,-1,1)
        expected=model(inputs_embeds=x);bounds=dict(start=0,acoustic_end=2)
        with patch_forward(model,model.blocks[0],x[:,:-ANSWER_TOKENS],bounds,bounds,'acoustic',torch):
            actual=model(inputs_embeds=x)
        torch.testing.assert_close(actual,expected)
        donor=x[:,:-ANSWER_TOKENS]+100
        source=model.blocks[0](torch.cat([donor,x[:,-ANSWER_TOKENS:]],1))[0]
        intermediate=model.blocks[0](x)[0].clone();intermediate[:,:2]=source[:,:2]
        expected=model.blocks[1](intermediate)[0]
        with patch_forward(model,model.blocks[0],donor,bounds,bounds,'acoustic',torch):
            actual=model(inputs_embeds=x)
        torch.testing.assert_close(actual,expected)

    def test_prefix_capture_stops_before_decoder_and_removes_hook(self):
        import torch
        model=self.model(torch);x=torch.ones(1,ANSWER_TOKENS+5,2);bounds=dict(start=1,acoustic_end=3)
        prefix,saved=capture_prefix(model,lambda:model(inputs_embeds=x),bounds)
        self.assertEqual(prefix.shape,(1,5,2));self.assertEqual(saved,bounds)
        self.assertEqual(model.seen,[]);self.assertFalse(model._forward_pre_hooks)

    def test_collection_does_not_patch_and_respects_original_forward_limit(self):
        import torch
        model=self.model(torch);x=torch.arange(ANSWER_TOKENS+3,dtype=torch.float32).reshape(1,-1,1)
        expected=model(inputs_embeds=x);bounds=dict(start=0,acoustic_end=2)
        donor=x[:,:-ANSWER_TOKENS]+100
        donor_hidden=model.blocks[0](torch.cat([donor,x[:,-ANSWER_TOKENS:]],1))[0][:,-ANSWER_TOKENS:]
        neutral_hidden=model.blocks[0](x)[0][:,-ANSWER_TOKENS:]
        with patch_forward(model,model.blocks[0],donor,bounds,bounds,'answer',torch,collect_only=True,max_forwards=1) as state:
            first=model(inputs_embeds=x);second=model(inputs_embeds=x)
        torch.testing.assert_close(first,expected);torch.testing.assert_close(second,expected)
        self.assertEqual((state['recipient_forwards'],state['donor_forwards'],state['patched_forwards']),(2,1,0))
        self.assertEqual(len(state['neutral_means']),1)
        torch.testing.assert_close(state['neutral_means'][0],neutral_hidden.mean(dim=(0,1)))
        torch.testing.assert_close(state['differences'][0],(donor_hidden-neutral_hidden).mean(dim=(0,1)))


if __name__=='__main__':unittest.main()
