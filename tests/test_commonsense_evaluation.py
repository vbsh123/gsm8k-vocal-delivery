from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from evaluate_commonsense_audio import main,score_choice,PROMPT


class CommonsenseEvaluationTests(unittest.TestCase):
    def test_explicit_choice_scoring_does_not_use_reasoning_or_gold_as_fallback(self):
        for text in ('Final answer: B','**Final answer:** (b).','Final answer:\nB','Final answer: option B'):
            self.assertTrue(score_choice(text,'B')['correct'])
        for text in ('I think B','Final answer: B or C','Final answer: bank','Final answer: 2','Final answer:'):
            self.assertTrue(score_choice(text,'B')['parse_failed'])
        self.assertFalse(score_choice('Final answer: A','B')['correct'])
        self.assertFalse(score_choice('Final answer: B\nCorrection. Final answer: C','B')['correct'])

    def test_freezes_audio_only_recipe_and_waits_for_complete_speed_queue(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            root=Path(d);source=root/'source';source.mkdir();output=root/'out'
            rows=[]
            for style in ('neutral','angry','happy','sad'):
                name=style+'.wav'
                with wave.open(str(source/name),'wb') as audio:
                    audio.setnchannels(1);audio.setsampwidth(2);audio.setframerate(16000);audio.writeframes(b'\0\0'*160)
                rows.append(dict(provider='azure',question_id='commonsense-q',voice='Jenny',style=style,audio_path=name,answer='D',question='Question? A. first. B. second. C. third. D. fourth. E. fifth.'))
            (source/'plan.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            argv=['test','--source',str(source),'--output',str(output),'--wait-for-speeds']
            with patch('sys.argv',argv+['--prepare-only']):main()
            config=json.loads((output/'run_config.json').read_text())
            self.assertEqual(len(config['recordings']),4)
            self.assertEqual(config['seed'],42)
            self.assertEqual(config['prompt'],PROMPT)
            with patch('sys.argv',argv),patch('evaluate_commonsense_audio.subprocess.run',side_effect=RuntimeError('wait sentinel')) as run:
                with self.assertRaisesRegex(RuntimeError,'wait sentinel'):main()
            self.assertEqual(run.call_args.args[0][-2:],['--kind','pitch-speed'])
            self.assertFalse((output/'results.jsonl').exists())


if __name__=='__main__':unittest.main()
