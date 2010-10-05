#@+leo-ver=4-thin
#@+node:gcross.20100923134429.1880:@thin test.py
#@<< Import needed modules >>
#@+node:gcross.20100923134429.1883:<< Import needed modules >>
import unittest
from paycheck import *
from numpy import array, zeros, all, double, tensordot, multiply, complex128, allclose, ones, diag, identity, dot, argmin, rank, set_printoptions
from numpy.linalg import norm
from numpy.random import rand
from random import randint, choice
import random
import __builtin__
from utils import *
#@-node:gcross.20100923134429.1883:<< Import needed modules >>
#@nl

#@+others
#@+node:gcross.20100923134429.1894:Classes
#@+node:gcross.20100923134429.1895:TestCase
class TestCase(unittest.TestCase):
    #@    @+others
    #@+node:gcross.20100923134429.1896:assertAllClose
    def assertAllClose(self,v1,v2):
        v1 = array(v1)
        v2 = array(v2)
        self.assertEqual(v1.shape,v2.shape)
        self.assertTrue(allclose(v1,v2))
    #@nonl
    #@-node:gcross.20100923134429.1896:assertAllClose
    #@+node:gcross.20100923134429.1897:assertAllEqual
    def assertAllEqual(self,v1,v2):
        v1 = array(v1)
        v2 = array(v2)
        self.assertEqual(v1.shape,v2.shape)
        self.assertTrue(all(v1 == v2))
    #@nonl
    #@-node:gcross.20100923134429.1897:assertAllEqual
    #@+node:gcross.20100923134429.1898:assertVanishing
    def assertVanishing(self,v):
        self.assertAlmostEqual(norm(v),0)
    #@-node:gcross.20100923134429.1898:assertVanishing
    #@-others
#@-node:gcross.20100923134429.1895:TestCase
#@-node:gcross.20100923134429.1894:Classes
#@+node:gcross.20100923134429.1886:Tests
#@+node:gcross.20100923134429.1888:matrix_vector_product
class matrix_vector_product(TestCase):
    contractor = staticmethod(
        form_contractor([
            ("M0","V'0"),
            ("M1","V0"),
        ], [
            ("M",2),
            ("V",1),
        ], ("V'",1)
        ))

    @with_checker(number_of_calls=10)
    def test(self,m=irange(1,10),n=irange(1,10)):
        M = crand(m,n)
        v = crand(n)
        self.assertAllClose(dot(M,v),self.contractor(M,v))


#@-node:gcross.20100923134429.1888:matrix_vector_product
#@-node:gcross.20100923134429.1886:Tests
#@-others


set_printoptions(linewidth=132)

tests = [
    matrix_vector_product,
    ]

#@<< Runner >>
#@+node:gcross.20100923134429.1885:<< Runner >>
unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(map(unittest.defaultTestLoader.loadTestsFromTestCase, tests)))
#@-node:gcross.20100923134429.1885:<< Runner >>
#@nl
#@nonl
#@-node:gcross.20100923134429.1880:@thin test.py
#@-leo
