import unittest
from unittest.mock import patch
import torch
import evaluate_bbh_candidates as run


class ConcurrentStartupTests(unittest.TestCase):
    def test_previous_dataset_cache_is_released_before_reading_free_memory(self):
        gib=2**30;events=[];free=[int(8.5*gib)]
        def release():events.append('release');free[0]=24*gib
        def memory(device):events.append('measure');return free[0],48*gib
        with patch('gc.collect',side_effect=lambda:events.append('collect')), \
             patch.object(torch.cuda,'empty_cache',side_effect=release), \
             patch.object(torch.cuda,'mem_get_info',side_effect=memory), \
             patch.object(torch.cuda,'set_per_process_memory_fraction') as cap:
            run.concurrent_budget_now(18.)
        self.assertEqual(events,['collect','release','measure'])
        cap.assert_called_once_with(18/48,device=0)

    def test_real_shortage_still_fails_without_waiting_or_changing_memory_cap(self):
        with patch('gc.collect'),patch.object(torch.cuda,'empty_cache'), \
             patch.object(torch.cuda,'mem_get_info',return_value=(int(8.5*2**30),48*2**30)), \
             patch.object(torch.cuda,'set_per_process_memory_fraction') as cap, \
             patch.object(run.time,'sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError,'have8.5'):run.concurrent_budget_now(18.)
        cap.assert_not_called();sleep.assert_not_called()


if __name__=='__main__':unittest.main()
