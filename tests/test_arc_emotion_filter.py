from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluate_arc_audio import report,STEERING
from run_arc_emotion2vec import main


@unittest.skipUnless(importlib.util.find_spec('numpy'),'Needs NumPy for confidence intervals')
class ArcEmotionFilterTests(unittest.TestCase):
    def test_pair_needs_both_matching_labels_and_steering_uses_same_neutral_input(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            root=Path(d);questions=[];rows=[];labels=[]
            for i in range(4):
                q=f'q{i}';questions.append(dict(question_id=q,answer='A',choices=[dict(label=k,text=k) for k in 'ABCD']))
                for style in ('neutral','angry'):
                    rows.append(dict(question_id=q,voice='Jenny',condition=style,style=style,audio_path=f'{q}-{style}.wav',
                        response='Final answer: A',reference_answer='A'))
                rows.append(dict(rows[-2],condition=STEERING))
            labels=[dict(audio_path='q0-neutral.wav',emotion='neutral'),dict(audio_path='q0-angry.wav',emotion='angry'),
                dict(audio_path='q1-neutral.wav',emotion='happy'),dict(audio_path='q1-angry.wav',emotion='angry'),
                dict(audio_path='q2-neutral.wav',emotion='neutral'),dict(audio_path='q2-angry.wav',emotion='neutral'),
                dict(audio_path='q3-neutral.wav',emotion='neutral')]
            (root/'questions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in questions))
            (root/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (root/'results-emotion2vec.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in labels))
            report(root,root)
            original=(root/'summary.json').read_bytes()
            report(root,root,summary_name='summary-emotion2vec.json',emotion2vec_agreed=True)
            filtered=json.loads((root/'summary-emotion2vec.json').read_text())
            by_condition={(r['condition'],r['metric']):r for r in filtered['paired']}
            self.assertEqual(by_condition['angry','answer_correct']['pairs'],1)
            self.assertEqual(by_condition[STEERING,'answer_correct']['pairs'],3)
            self.assertEqual((root/'summary.json').read_bytes(),original)
            self.assertTrue(filtered['emotion2vec_agreed'])

    def test_missing_classification_is_reported_instead_of_zero_accuracy(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaisesRegex(ValueError,'predictions missing'):
                report(root,root,rows=[],emotion2vec_agreed=True)


class ArcClassificationQueueTests(unittest.TestCase):
    def test_queue_only_classifies_and_summarizes_without_accuracy_generation(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            root=Path(d);source=root/'source';source.mkdir();(source/'plan.jsonl').write_text('')
            (root/'.venv-emotion/bin').mkdir(parents=True);(root/'.venv-emotion/bin/python').touch()
            (root/'outputs').mkdir();old=Path.cwd()
            try:
                os.chdir(root)
                with patch('sys.argv',['test','--source',str(source)]),patch('run_arc_emotion2vec.subprocess.run') as calls,patch('run_arc_emotion2vec.subprocess.check_output',return_value=''):
                    main()
                commands=[c.args[0] for c in calls.call_args_list]
                classify=[c for c in commands if 'classify_cloud_smoke.py' in c]
                self.assertEqual(len(classify),1);self.assertIn('emotion2vec',classify[0])
                evaluations=[c for c in commands if 'evaluate_arc_audio.py' in c]
                self.assertEqual(len(evaluations),2)
                self.assertTrue(all('--summary-only' in c for c in evaluations))
                self.assertIn('--emotion2vec-agreed',evaluations[1])
            finally:os.chdir(old)


if __name__=='__main__':unittest.main()
