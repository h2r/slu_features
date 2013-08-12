from spatial_features_cxx import spatial_features_names_avs_polygon_polygon, \
    spatial_features_avs_polygon_polygon
import unittest
import numpy as na
from numpy import transpose as tp

from numpy import array
from features.feature_utils import compute_fdict
import pylab as mpl
from scipy import allclose

class TestCase(unittest.TestCase):

    def destPolygons(self):
        fvec = spatial_features_avs_polygon_polygon(tp([(0, 0), (1, 0),
                                                           (1, 1), (0, 1)]),
                                                       tp([(2, 0), (3, 0),
                                                           (3, 1), (2, 1)]),
                                                       0)


        self.assertFalse(any(na.isnan(x) for x in fvec))



        fvec = spatial_features_avs_polygon_polygon(tp([(0, 0), (1, 0),
                                                           (1, 1), (0, 1)]),
                                                       tp([(0, 0), (1, 0),
                                                           (1, 1), (0, 1)]),
                                                       0)
        self.assertFalse(any(na.isnan(x) for x in fvec))

    def testFunkyArguments(self):
        """
        Test that it returns the same values. 
        """
        otheta = 1.1752771158
        of_points_xy = [[ 19.08576266, 20.43434719, 19.77759778, 18.42901326,],
                        [ 34.78776701, 35.44451641, 36.79310093, 36.13635153,]]
        ol_points_xy = [[ 19.08576266, 20.43434719, 19.77759778, 18.42901326,],
                        [ 34.78776701, 35.44451641, 36.79310093, 36.13635153,]]

        result = None

        for i in range(0, 20):
            print "i", i
            fvec = compute_fdict(spatial_features_names_avs_polygon_polygon(),
                                 spatial_features_avs_polygon_polygon(of_points_xy,
                                                                         ol_points_xy,
                                                                         otheta))
            if result == None:
                result = fvec
            else:
                for key in result.keys():
                    print key
                    self.assertEqual(result[key], fvec[key])


