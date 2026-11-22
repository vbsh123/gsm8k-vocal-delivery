import unittest

from diffa_gsm_shared_emotions import split_ids


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


if __name__ == '__main__':
    unittest.main()
