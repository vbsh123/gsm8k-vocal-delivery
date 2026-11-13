"""Wait for parent queues as well as workers, scoped to the current repository."""

from pathlib import Path
import tempfile
import unittest

from wait_for_experiments import active_jobs


class WaitTests(unittest.TestCase):
    def test_waits_for_all_matching_parents_and_workers_without_matching_log_viewers(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'repo'; root.mkdir()
            other=Path(directory)/'other'; other.mkdir()
            proc=Path(directory)/'proc'; proc.mkdir()
            for pid,args,cwd,state in [
                (10,['bash','run_dataset_emotions.sh','cuda:0'],root,'S'),
                (11,['python','classify_dataset_emotions.py','--dataset','mmau'],root,'R'),
                (12,['tail','-f','classification.log'],root,'S'),
                (13,['python','classify_dataset_emotions.py'],other,'R'),
                (14,['python','steer_audio.py'],root,'R'),
                (15,['python','classify_dataset_emotions.py'],root,'Z'),
                (16,['python','run_azure_followups.py','--stage','classify'],root,'S'),
                (17,['python','run_azure_followups.py','--stage','steering'],root,'S'),
                (18,['bash','run_emotions_after_steering.sh'],root,'S'),
                (19,['bash','run_jenny_third_accuracy.sh'],root,'S'),
                (20,['python','run_azure_followups.py','--stage','diffa'],root,'S'),
                (21,['python','run_jenny_late_layers.py'],root,'S'),
                (22,['python','finish_jenny_finalists.py'],root,'S'),
                (23,['python','run_L31_forward20_after_current.py'],root,'S'),
                (24,['python','collect_temporal_vectors.py'],root,'S'),
                (25,['bash','-c','bash setup_qwen_omni.sh && python run_azure_followups.py --stage qwen && bash setup_kimi_audio.sh'],root,'S'),
                (26,['python','evaluate_kimi_audio.py'],root,'R'),
                (27,['python','run_azure_followups.py','--stage','qwen'],root,'S'),
                (28,['python','run_pitch_speed_after_models.py'],root,'S'),
                (29,['python','evaluate_audio_speed.py'],root,'R'),
                (30,['python','evaluate_audio_pitch.py'],root,'R'),
                (31,['python','evaluate_commonsense_audio.py','--wait-for-speeds'],root,'S')]:
                folder=proc/str(pid); folder.mkdir()
                (folder/'cwd').symlink_to(cwd,target_is_directory=True)
                (folder/'cmdline').write_bytes(b'\0'.join(a.encode() for a in args))
                (folder/'stat').write_text(f'{pid} (python) {state} 0 0')
            self.assertEqual(active_jobs('classification',root,proc),[10,11,16])
            self.assertEqual(active_jobs('steering',root,proc),[14,17])
            self.assertEqual(active_jobs('azure-emotions',root,proc),[16,18,19,20])
            self.assertEqual(active_jobs('finalist-queue',root,proc),[14,17,21,22,24])
            self.assertEqual(active_jobs('audio-models',root,proc),[25,26,27])
            self.assertEqual(active_jobs('pitch-speed',root,proc),[28,29,30])


if __name__=='__main__':
    unittest.main()
