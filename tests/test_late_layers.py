import unittest

from run_jenny_late_layers import conditions


class LateLayerConditionsTest(unittest.TestCase):
    def test_complete_individual_sweep(self):
        rows = conditions()
        self.assertEqual(rows[0], dict(name='baseline', interventions=[]))
        self.assertEqual(len(rows), 25)
        self.assertEqual(len({r['name'] for r in rows}), 25)
        actual = set()
        for row in rows[1:]:
            self.assertEqual(len(row['interventions']), 1)
            target = row['interventions'][0]
            actual.add((target['layer'], target['group'], target['strength'], row['schedule']))
        expected = {(l, g, a, s) for l in (29, 30, 31)
                    for g in ('acoustic', 'answer') for a in (0.5, 1)
                    for s in ('always', 'periodic2')}
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
