from stringprep import b3_exceptions
import numpy as np
from backend import forwardSubstitution, backwardSubstitution, rowrank
from unittest import TestCase

class MyTests(TestCase):
    def test_solveLinearSystem(self):
        A = np.array([[0,-3,-6,4],[-1,-2,-1,3],[-2,-3,0,3],[1,4,5,-9]], dtype=np.float32)
        b = np.array([9,1,-1,-7], dtype=np.float32)
        A_expected = np.array([[1,4,5,-9],[0,2,4,-6],[0,0,0,-5],[0,0,0,0]], dtype=np.float32)
        b_expected = np.array([-7,-6,0,0], dtype=np.float32)
        A_actual, b_actual = forwardSubstitution(A, b)
        self.assertTrue(np.array_equal(A_expected, A_actual))
        self.assertTrue(np.array_equal(b_expected, b_actual))

    def test_backwards_substitution(self):
        A = np.array([[-2,-3,0,3],[0, -3,-6,4],[0,0,0,-4.16],[0,0,0,0]], dtype=np.float32)
        b = np.array([-1,9,0,0], dtype=np.float32)
        backwardSubstitution(A, b)

    def test_rowrank(self):
        A = np.array([[0,-3,-6,4],[-1,-2,-1,3],[-2,-3,0,3],[1,4,5,-9]], dtype=np.float32)
        expected = 3
        actual = rowrank(A)
        self.assertEqual(expected, actual)



t = MyTests()
# t.test_solveLinearSystem()
t.test_backwards_substitution()


        

