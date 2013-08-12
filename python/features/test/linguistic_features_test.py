import unittest
from features import linguistic_features as lf
        
class TestCase(unittest.TestCase):

    def testDepluralize(self):
        self.assertEqual(lf.depluralize("tire"),
                         "tire")

        self.assertEqual(lf.depluralize("tires"),
                         "tire")


        self.assertEqual(lf.depluralize("pallets"),
                         "pallet")

        self.assertEqual(lf.depluralize("boxes"),
                         "box")
        self.assertEqual(lf.depluralize("to"),
                         "to")   


        self.assertEqual(lf.depluralize("a"),
                         "a")   


        self.assertEqual(lf.depluralize("nonword"),
                         "nonword")   
        
    def testLinguisticFeatures(self):

        featureDict = lf.sfe_language_object(["tire"], [], ["pallet"])
        self.assertTrue(featureDict != None)
        
