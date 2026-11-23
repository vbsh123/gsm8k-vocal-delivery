from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
import audit_diffa_vanilla_steering as audit
from diffa_gsm_emotion import capture
from diffa_stream_collection import append, finalize
from extract_audio_vectors import average_pair
from steer import install_interventions


class VectorAuditTests(unittest.TestCase):
    def model(self):
        class Block(torch.nn.Module):
            def forward(self, hidden):
                return hidden + 1, 'cache-preserved'
        blocks = torch.nn.ModuleList([Block() for _ in range(32)])
        model = SimpleNamespace(config=SimpleNamespace(prompt_size=2),
            llm_model=SimpleNamespace(model=SimpleNamespace(transformer=SimpleNamespace(blocks=blocks))))
        decoder = SimpleNamespace(insert_audio_embedding=lambda text, speech, lengths:
            torch.cat([text[:, :int(lengths[0])], speech, text[:, int(lengths[0]):]], dim=1))
        return model, blocks, decoder

    def test_actual_capture_aggregate_load_and_hook_round_trip(self):
        model, blocks, decoder = self.model()
        original_insert = decoder.insert_audio_embedding
        def generate(offset, forwards):
            # Two prompt slots, four audio slots, eight answer slots.
            hidden = decoder.insert_audio_embedding(torch.zeros(1,10,3), torch.zeros(1,4,3), torch.tensor([1]))
            for step in range(forwards):
                h = hidden.clone()
                h[:, -8:] = offset + step
                for block in blocks:
                    h, cache = block(h)
                    self.assertEqual(cache, 'cache-preserved')
            return f'response-{offset}', {}
        nr, ne, n = capture(model, lambda: generate(0., 2), decoder, 8)
        er, ee, e = capture(model, lambda: generate(3., 5), decoder, 8)
        self.assertEqual(n.shape, (32,2,4,3))
        self.assertEqual(e.shape, (32,5,4,3))
        torch.testing.assert_close(n[0,:,3,0], torch.tensor([1.,2.]))
        torch.testing.assert_close(n[31,:,3,0], torch.tensor([32.,33.]))
        # Audio groups observe actual audio positions, not the answer tail.
        torch.testing.assert_close(n[:,0,:3,0], torch.arange(1.,33.).unsqueeze(1).expand(32,3))
        nm, em, delta, matched = average_pair(n,e,'matched')
        torch.testing.assert_close(delta[:,3], torch.full((32,3),3.))
        torch.testing.assert_close(delta[:,:3], torch.zeros(32,3,3))
        self.assertEqual(matched,2)  # Three later emotional forwards are deliberately excluded.
        state = dict(completed_ids=[],rows=[],difference_sum=None,unit_sum=None,valid_count=None)
        append(state,dict(question_id='fit',neutral=nm,emotion=em,difference=delta,
            neutral_response=nr,emotion_response=er,neutral_forwards=2,emotion_forwards=5,
            matched_forwards=2,neutral_encoding=ne,emotion_encoding=ee))
        plan = dict(protocol='test',pairs=[dict(question_id='fit',neutral={},emotion={})],
            emotion='angry',collection_prompt='prompt',max_new_tokens=8,collection_ids=['fit'],averaging='matched')
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            finalize(plan,folder,state)
            loaded,_=audit.vanilla.task.winners.read_vector(folder/'style-vectors.pt','angry',28,'answer')
            torch.testing.assert_close(loaded,torch.full((3,),3.))
            for strength in (-1.,0.,1.):
                hidden=torch.zeros(1,14,3)
                original=hidden.clone()
                interventions=[dict(layer=28,group='answer',strength=strength)] if strength else []
                with install_interventions(blocks,{(28,'answer'):loaded},interventions,{},8) as states:
                    result,cache=blocks[28](hidden)
                torch.testing.assert_close(hidden,original)
                torch.testing.assert_close(result[:,:6],torch.ones(1,6,3))
                torch.testing.assert_close(result[:,-8:],torch.full((1,8,3),1.+3.*strength))
                self.assertEqual(cache,'cache-preserved')
        self.assertIs(decoder.insert_audio_embedding,original_insert)
        self.assertTrue(all(not block._forward_hooks for block in blocks))

    def test_capture_cleans_up_on_decoder_failure(self):
        model, blocks, decoder=self.model()
        original=decoder.insert_audio_embedding
        def fail():
            raise RuntimeError('interrupted')
        with self.assertRaisesRegex(RuntimeError,'interrupted'):
            capture(model,fail,decoder,8)
        self.assertIs(decoder.insert_audio_embedding,original)
        self.assertTrue(all(not block._forward_hooks for block in blocks))

    def fixture(self, root):
        folder=root/'angry/agreed'
        folder.mkdir(parents=True)
        plan=dict(collection_ids=['qfit'],collection_prompt='prompt',prompt='prompt',max_new_tokens=512)
        config=folder/'run_config.json'
        config.write_text(json.dumps(plan))
        delta=torch.full((32,4,3),4.)
        state=dict(run_config_sha256=audit.vanilla.task.winners.digest(config),completed_ids=['qfit'],
            difference_sum=delta.double(),rows=[dict(question_id='qfit',neutral_response='#### 1',
                emotion_response='#### 2',neutral_forwards=2,emotion_forwards=10,matched_forwards=2)])
        torch.save(state,folder/'collection-stream.pt')
        vector=dict(sign='angry minus neutral',prompt='prompt',max_new_tokens=512,
            groups=['acoustic','semantic','combined','answer'],extraction_question_ids=['qfit'],
            pair_mean_sums=dict(angry=delta),pair_counts=dict(angry=torch.ones(32,dtype=torch.long)))
        torch.save(vector,folder/'style-vectors.pt')
        delivery=root/'delivery'
        delivery.mkdir()
        rows=[dict(question_id='qfit',condition=style,response=response,prompt='prompt',strength=0.,interventions=[])
            for style,response in [('neutral','#### 1'),('angry','#### 2')]]
        (delivery/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return folder

    def test_saved_audit_reports_truncation_and_rejects_changed_vector(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            folder=self.fixture(root)
            with redirect_stdout(io.StringIO()):
                result=audit.audit(root,'angry','agreed')
            self.assertEqual(result['mean_emotion_forward_retained_pct'],20.)
            self.assertEqual(result['compared_responses'],2)
            path=folder/'style-vectors.pt'
            vector=torch.load(path,weights_only=True)
            vector['pair_mean_sums']['angry']/=2
            torch.save(vector,path)
            with self.assertRaisesRegex(ValueError,'sum differs'):
                audit.audit(root,'angry','agreed')

    def test_saved_audit_rejects_collection_natural_response_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.fixture(root)
            path=root/'delivery/results.jsonl'
            path.write_text(path.read_text().replace('#### 2','#### 999'))
            with self.assertRaisesRegex(ValueError,'response mismatch'):
                audit.audit(root,'angry','agreed')


if __name__=='__main__':
    unittest.main()
