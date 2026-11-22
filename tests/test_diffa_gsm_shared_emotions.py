import unittest

from diffa_gsm_shared_emotions import PROTOCOL, split_ids, validate_protocol
from copy import deepcopy


class SharedGsmSplit(unittest.TestCase):
    def test_all_shared_when_less_than_half(self):
        train, test = split_ids(range(1269), range(609))
        self.assertEqual(len(train), 609)
        self.assertEqual(len(test), 660)
        self.assertFalse(set(train) & set(test))

    def test_half_cap_and_reproducibility(self):
        train, test = split_ids(range(1000), range(609))
        self.assertEqual(len(train), 500)
        self.assertEqual(len(test), 500)
        self.assertEqual((train, test), split_ids(range(1000), reversed(range(609))))
        self.assertTrue(set(train) <= set(range(609)))

    def test_missing_neutral_rejected(self):
        with self.assertRaises(ValueError):
            split_ids(range(10), range(11))

    def test_reject_prompt_mismatch_and_old_protocol(self):
        prompt = 'Zero-shot audio prompt ending #### <number>.'
        split = dict(protocol=PROTOCOL, neutral_pool_ids=[1, 2], collection_ids=[1], test_ids=[2])
        plan = dict(protocol=PROTOCOL, collection_ids=[1], test_ids=[2],
                    prompt=prompt, collection_prompt=prompt, target=dict(prompt=prompt))
        plans = {emotion: deepcopy(plan) for emotion in ('happy', 'angry')}
        validate_protocol(split, plans, prompt)
        plans['happy']['collection_prompt'] = 'Final answer: <number>'
        with self.assertRaisesRegex(ValueError, 'zero-shot'):
            validate_protocol(split, plans, prompt)
        with self.assertRaisesRegex(ValueError, 'another prompt protocol'):
            validate_protocol(dict(split, protocol='old'), plans, prompt)


if __name__ == '__main__':
    unittest.main()
