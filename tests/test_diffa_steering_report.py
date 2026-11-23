from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import diffa_happy_nightly as report


class SteeringReportTests(unittest.TestCase):
    def run_report(self, rows, loader):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder/'results.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            def score(_):
                (folder/'summary.json').write_text(json.dumps(dict(paired=['accuracy retained'])))
            with patch.object(report.winners, 'report_job', side_effect=score), patch.dict('sys.modules',
                    {'transformers': SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=loader))}), patch(
                    'diffa_model_receipts.model_revisions', return_value={'GSAI-ML/LLaDA-8B-Instruct': 'a'*40}), redirect_stdout(io.StringIO()):
                report.report_steering(folder)
            return json.loads((folder/'summary.json').read_text())

    def row(self, condition, tokens):
        return dict(question_id='q0', voice='Jenny', condition=condition, response='#### 2', response_tokens=tokens)

    def test_baseline_only_report_never_loads_tokenizer(self):
        loader = Mock(side_effect=AssertionError('Must not load tokenizer'))
        result = self.run_report([self.row('neutral', 3)], loader)
        loader.assert_not_called()
        self.assertEqual(result['paired'], ['accuracy retained'])
        self.assertEqual(result['lengths'], {})

    def test_saved_token_counts_need_no_tokenizer(self):
        loader = Mock(side_effect=AssertionError('Must not load tokenizer'))
        result = self.run_report([self.row('neutral', 3), self.row('L28-answer-a1', 10)], loader)
        loader.assert_not_called()
        self.assertEqual(result['lengths']['neutral']['mean_tokens'], 3)
        self.assertEqual(result['lengths']['L28-answer-a1']['mean_tokens'], 10)

    def test_missing_local_tokenizer_keeps_accuracy_and_uses_pinned_revision(self):
        loader = Mock(side_effect=OSError('not cached'))
        result = self.run_report([self.row('neutral', 3), self.row('L28-answer-a1', None)], loader)
        loader.assert_called_once_with('GSAI-ML/LLaDA-8B-Instruct', revision='a'*40, local_files_only=True)
        self.assertEqual(result['paired'], ['accuracy retained'])
        self.assertEqual(result['lengths'], {})
        self.assertIn('Accuracy scores remain valid', result['lengths_note'])


if __name__ == '__main__':
    unittest.main()
