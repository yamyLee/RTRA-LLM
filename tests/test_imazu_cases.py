import unittest

from src.utils.imazu_cases import get_case_numbers, get_obstacle_data


class ImazuCasesTest(unittest.TestCase):
    def test_supports_23_cases(self):
        self.assertEqual(get_case_numbers(), tuple(range(1, 24)))

    def test_case_23_has_obstacle_data(self):
        x_ob, y_ob, v_ob, psi_ob = get_obstacle_data(23)

        self.assertEqual(len(x_ob), 1)
        self.assertEqual(len(y_ob), 1)
        self.assertEqual(len(v_ob), 1)
        self.assertEqual(len(psi_ob), 1)

    def test_invalid_case_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Unsupported case_number"):
            get_obstacle_data(24)


if __name__ == "__main__":
    unittest.main()
