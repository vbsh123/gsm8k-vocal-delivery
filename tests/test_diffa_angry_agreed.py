from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
import wave
from unittest.mock import patch

import diffa_angry_agreed as run


def fixture():
    pinned = {}; labels = {}
    for q in ['yes', 'no', 'test']:
        for s in ['neutral', 'angry']:
            key = q+s
            record = dict(question_id=q, voice='en-US-JennyNeural', audio_sha256=key,
                          audio_file='/audio/'+key+'.wav', reference_answer='2')
            pinned[q+'|'+s] = dict(record=record, question='question '+q)
            labels[key] = dict(clip_id=key, audio_sha256=key, model=run.MODEL,
                               classification_status='classified', emotion='neutral' if q == 'no' else s)
    target = dict(name='svamp', task='svamp', run='/original', bank={'test': dict(answer='2')},
                  prompt='#### <number>', vector_info=dict(collection_prompt='collection prompt'),
                  max_new_tokens=512, records=[pinned['test|neutral']['record']], extraction_ids=['yes', 'no'])
    return dict(target=target, collection_ids=['yes', 'no'], test_ids=['test'], pinned=pinned, recordings=[]), labels


class AngryAgreedTests(unittest.TestCase):
    def test_happy_gate_requires_complete_labels_and_strictly_more_than_half(self):
        data, labels = fixture()
        for q in data['collection_ids']:
            data['pinned'][q+'|happy'] = deepcopy(data['pinned'][q+'|angry'])
            data['pinned'][q+'|happy']['record']['audio_sha256'] = q+'happy'
            labels[q+'happy'] = dict(classification_status='classified', emotion='happy')
        with patch('builtins.print'):
            result = run.happy_agreement_gate({'svamp': data}, labels, ['svamp'], 50)
            self.assertEqual(result['svamp']['percent'], 100.)
            labels['nohappy']['emotion'] = 'neutral'
            with self.assertRaisesRegex(ValueError, 'strictly above 50'):
                run.happy_agreement_gate({'svamp': data}, labels, ['svamp'], 50)
            labels['nohappy']['emotion'] = 'happy'; labels['nohappy']['classification_status'] = 'error'
            with self.assertRaisesRegex(ValueError, 'Not ready'):
                run.happy_agreement_gate({'svamp': data}, labels, ['svamp'], 50)

    def test_inventory_and_summary_use_frozen_halves_and_report_pair_eligibility(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); audio = root/'audio'; audio.mkdir(); rows = []
            records = {}
            for i, q in enumerate(['a', 'b', 'test']):
                for j, style in enumerate(run.STYLES):
                    path = audio/f'{q}-{style}.wav'
                    with wave.open(str(path), 'wb') as f:
                        f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
                        f.writeframes(bytes([i*3+j, 0])*1600)
                    row = dict(question_id=q, question='question '+q, voice='en-US-JennyNeural',
                               provider='azure', style=style, audio_path=path.name, answer='2')
                    rows.append(row); records[q, style] = run.collection.make_record(row, audio)
            (audio/'plan.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            config = root/'original.json'
            config.write_text(json.dumps(dict(extraction=[dict(question_id=q, neutral=records[q, 'neutral'], angry=records[q, 'angry'])
                                                       for q in ['a', 'b']], recordings=[])))
            target = dict(config_file=str(config), records=[records['test', 'neutral']], extraction_ids=['a', 'b'],
                          vector_info=dict(pairs=2))
            out = root/'audit'
            with patch.object(run.winners, 'load_dataset', return_value=target), patch.object(run, 'seed_labels'):
                cohort = run.inventory(out, ['svamp'], {'svamp': {}}, run.audio_candidates(root))
            clips = run.read_rows(out/'audio_inventory.jsonl')
            self.assertEqual(len(clips), 9)
            labels = []
            for clip in clips:
                alias = clip['aliases'][0]
                emotion = alias['requested_style']
                if alias['question_id'] == 'b' and emotion == 'angry': emotion = 'neutral'
                labels.append(dict(clip_id=clip['clip_id'], classification_status='classified', emotion=emotion))
            (out/'results-emotion2vec.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in labels))
            with patch('builtins.print'): summary = run.summarize_audit(out)
            eligibility=summary['eligibility']['svamp']
            self.assertEqual(eligibility['collection_pool'],2)
            self.assertEqual(eligibility['agreed_pairs'],1)
            self.assertEqual(eligibility['test_questions'],1)
            self.assertEqual(eligibility['happy_agreed_pairs'],2)
            self.assertEqual(eligibility['happy_agreed_pct'],100.)
            self.assertEqual(eligibility['shared_angry_happy_pairs'],1)
            self.assertEqual(cohort['datasets']['svamp']['test_ids'], ['test'])
            self.assertTrue((out/'dataset-summary.csv').is_file())

    def test_selection_requires_both_labels_and_never_borrows_test(self):
        data, labels = fixture()
        pairs = run.agreed_pairs(data, labels)
        self.assertEqual([p['question_id'] for p in pairs], ['yes'])
        labels['yesneutral']['emotion'] = 'happy'
        self.assertEqual(run.agreed_pairs(data, labels), [])

    def test_filtered_plan_preserves_target_recipe_and_actual_pair_count(self):
        data, labels = fixture(); original = deepcopy(data)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root/'cohorts.json').write_text('{}')
            plan = run.prepare_plan('svamp', data, labels, root, root/'new')
            self.assertEqual(plan['collection_ids'], ['yes'])
            self.assertEqual(plan['collection_pool_ids'], ['yes', 'no'])
            self.assertEqual(plan['test_ids'], ['test'])
            self.assertEqual(plan['target'], original['target'])
            self.assertEqual(plan['max_new_tokens'], 512)
            self.assertEqual(plan['collection_prompt'], 'collection prompt')
            self.assertEqual(plan['prompt'], '#### <number>')
            self.assertEqual(plan['emotion'], 'angry')
            self.assertEqual(data, original)
            labels['noangry']['classification_status'] = 'error'
            with self.assertRaisesRegex(ValueError, 'pending/error'):
                run.prepare_plan('svamp', data, labels, root, root/'bad')

    def test_classified_failure_is_not_treated_as_disagreement(self):
        data, labels = fixture(); labels.pop('noneutral')
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root/'cohorts.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'pending/error'):
                run.prepare_plan('svamp', data, labels, root, root/'new')

    def test_transfer_overlap_rejected(self):
        source = dict(extraction_ids=['q'], vector='vector.pt', vector_sha256='hash')
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            run.job_for(source, dict(records=[dict(question_id='q')]), [0., 1.])

    def test_classifier_cache_reuse_requires_exact_hash_model_and_whole_recording(self):
        with tempfile.TemporaryDirectory() as d:
            previous = Path.cwd()
            try:
                os.chdir(d); root = Path('outputs/new'); root.mkdir(parents=True)
                (root/'audio_inventory.jsonl').write_text(json.dumps(dict(clip_id='good'))+'\n')
                old = Path('outputs/old'); old.mkdir()
                row = dict(clip_id='legacy', audio_sha256='good', model=run.MODEL,
                           processing_mode='whole_recording', classification_status='classified', emotion='angry')
                (old/'results-emotion2vec.jsonl').write_text('\n'.join(json.dumps(r) for r in [
                    dict(row, model='wrong'), dict(row, processing_mode='chunked'), row]))
                run.seed_labels(root)
                rows = run.read_rows(root/'results-emotion2vec.jsonl')
                self.assertEqual(len(rows), 1); self.assertEqual(rows[0]['clip_id'], 'good')
                self.assertIn('reused_from', rows[0])
            finally: os.chdir(previous)

    def test_strength_reuses_same_filtered_vector_but_not_another_direction(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); old = root/'primary'; old.mkdir(); new = root/'strengths'; new.mkdir()
            vector = root/'vector.pt'; vector.write_bytes(b'filtered vector')
            record = dict(question_id='q', voice='en-US-JennyNeural', audio_sha256='audio', reference_answer='2')
            target = dict(records=[record], run=str(old), reuse_runs=[], prompt='prompt', seed=42,
                          decoder='fast_dllm', max_new_tokens=512, steps=128, block_length=32, temperature=0.,
                          own_condition='L31-answer-a1')
            source = dict(extraction_ids=['train'], vector=str(vector), vector_sha256=run.winners.digest(vector), name='svamp')
            target['name'] = 'svamp'; target['task'] = 'svamp'
            job = run.job_for(source, target, [0., 1., 2.])
            (old/'run_config.json').write_text(json.dumps(job))
            rows = [dict(record, condition=c, strength=a, response='#### 2', prompt='prompt', seed=42,
                         decoder='fast_dllm', max_new_tokens=512, steps=128, block_length=32, temperature=0.,
                         interventions=[dict(layer=31, group='answer', strength=a)] if a else [])
                    for c, a in [('neutral', 0.), ('L31-answer-a1', 1.)]]
            (old/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            run.winners.reuse_answers(job, new)
            self.assertEqual(len(run.read_rows(new/'results.jsonl')), 2)
            other = root/'other'; other.mkdir(); job['source'] = dict(source, vector_sha256='different')
            run.winners.reuse_answers(job, other)
            self.assertEqual([r['condition'] for r in run.read_rows(other/'results.jsonl')], ['neutral'])


if __name__ == '__main__': unittest.main()
