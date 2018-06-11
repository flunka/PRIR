import unittest
import kmp


class TestKMP(unittest.TestCase):

    def test_build_partial_match_table(self):
        pattern = "ABCDABD"
        pm_table = [-1, 0, 0, 0, 0, 1, 2]
        result = kmp.build_partial_match_table(pattern)
        self.assertListEqual(list(result), pm_table)

        pattern = "ABACABABC"
        pm_table = [-1, 0, 0, 1, 0, 1, 2, 3, 2]
        result = kmp.build_partial_match_table(pattern)
        self.assertListEqual(list(result), pm_table)

        pattern = "PARTICIPATE IN PARACHUTE"
        pm_table = [-1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0]
        result = kmp.build_partial_match_table(pattern)
        self.assertListEqual(list(result), pm_table)

    def test_get_number_of_occurrence(self):
        pattern = "lo"
        filepath = "./test1"
        result = kmp.get_number_of_occurrence(pattern, filepath)
        self.assertEqual(result, 2)

        pattern = "tempor fermentum"
        filepath = "./test2"
        result = kmp.get_number_of_occurrence(pattern, filepath)
        self.assertEqual(result, 1)

        # pattern = "lo"
        # filepath = "./test3.test"
        # result = kmp.get_number_of_occurrence(pattern, filepath)
        # self.assertEqual(result, 3200000)


if __name__ == '__main__':
    unittest.main()
