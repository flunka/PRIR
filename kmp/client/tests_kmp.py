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

    def test_build_patterns(self):
        expression = "abc"
        patterns = ["abc"]
        result = kmp.build_patterns(expression)
        for pattern in patterns:
            self.assertEqual(next(result), pattern)

        expression = "a[bc]d"
        patterns = ["abd", "acd"]
        result = kmp.build_patterns(expression)
        for pattern in patterns:
            self.assertEqual(next(result), pattern)

        expression = "[ab]c[def]"
        patterns = ["acd", "bcd", "ace", "bce", "acf", "bcf"]
        result = kmp.build_patterns(expression)
        for pattern in patterns:
            self.assertEqual(next(result), pattern)

        expression = "Hello World!?.()#$% ^\n&*"
        patterns = ["Hello World!?.()#$% ^\n&*"]
        result = kmp.build_patterns(expression)
        for pattern in patterns:
            self.assertEqual(next(result), pattern)

        expression = "]abc[def"
        with self.assertRaises(ValueError):
            list(kmp.build_patterns(expression))

        expression = "[abc\\d]def"
        with self.assertRaises(ValueError):
            list(kmp.build_patterns(expression))

        expression = "\\d"
        patterns = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        result = kmp.build_patterns(expression)
        for pattern in patterns:
            self.assertEqual(next(result), pattern)

        expression = "[ab]\\d"
        patterns = ["a0", "b0", "a1", "b1", "a2", "b2", "a3", "b3", "a4", "b4", "a5", "b5", "a6", "b6", "a7", "b7", "a8", "b8", "a9", "b9"]
        result = kmp.build_patterns(expression)
        for pattern in patterns:
            self.assertEqual(next(result), pattern)


if __name__ == '__main__':
    unittest.main()
