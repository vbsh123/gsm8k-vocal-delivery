import json
from pathlib import Path
import tempfile
import unittest

import torch
from compare_diffa_emotion_vectors import geometry,load_vector,compare,direction


class EmotionVectorGeometryTests(unittest.TestCase):
    def test_direction_scale_and_orthogonal_residual_are_distinct(self):
        a=torch.tensor([1.,0.])
        scaled=geometry(a,2*a)
        self.assertEqual(scaled['cosine'],1.)
        self.assertEqual(scaled['happy_over_angry_norm'],2.)
        self.assertEqual(scaled['relative_distance_to_angry'],1.)
        self.assertEqual(scaled['happy_orthogonal_squared_fraction'],0.)
        perpendicular=geometry(a,torch.tensor([0.,1.]))
        self.assertEqual(perpendicular['cosine'],0.)
        self.assertEqual(perpendicular['angle_degrees'],90.)
        self.assertEqual(perpendicular['happy_orthogonal_squared_fraction'],1.)
        self.assertEqual(geometry(a,-a)['cosine'],-1.)
        self.assertIsNone(geometry(a,torch.zeros(2))['cosine'])

    def test_actual_fitting_ids_and_uncollected_targets_and_consistency(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data={}
            for e,value in [('angry',1.),('happy',2.)]:
                folder=root/e;folder.mkdir()
                sums=torch.zeros(32,2,3);sums[31,1,0]=2*value
                counts=torch.zeros(32,dtype=torch.long);counts[31]=2
                state=dict(pair_mean_sums={e:sums},pair_counts={e:counts},groups=['semantic','answer'],
                           collected_targets=[(31,'answer')],prompt='same',max_new_tokens=512,seed=42,
                           extraction_question_ids=['a','b','excluded'],pairs=[dict(question_id=q) for q in ('a','b')])
                path=folder/'style-vectors.pt';torch.save(state,path)
                data[e]=load_vector(path,e)
                unit=torch.zeros(32,2,3,dtype=torch.float64);unit[31,1,0]=2.
                valid=torch.zeros(32,2,dtype=torch.long);valid[31,1]=2
                torch.save(dict(completed_ids=['a','b'],unit_sum=unit,valid_count=valid),folder/'collection-stream.pt')
            self.assertEqual(data['angry']['ids'],['a','b'])
            self.assertIsNone(direction(data['angry'],31,'semantic'))
            report=compare(data['angry'],data['happy'],root/'angry',root/'happy')
            self.assertTrue(report['same_fitting_questions'])
            self.assertEqual(len(report['rows']),1)
            row=report['rows'][0]
            self.assertEqual(row['cosine'],1.)
            self.assertEqual(row['happy_over_angry_norm'],2.)
            self.assertEqual(row['angry_direction_consistency'],1.)
            data['happy']['state']['prompt']='different'
            self.assertFalse(compare(data['angry'],data['happy'],root/'angry',root/'happy')['same_collection_prompt'])


if __name__=='__main__':unittest.main()
