import unittest

from src.risk_assessment.risk_calculations import risk_calculations


class RiskCalculationsTest(unittest.TestCase):
    def test_close_future_encounter_has_high_risk(self):
        risk = float(risk_calculations(dcpa=100.0, tcpa=60.0, distance_ob=100.0, v_rel=5.0))

        self.assertGreater(risk, 0.9)

    def test_far_encounter_has_low_risk(self):
        risk = float(risk_calculations(dcpa=1200.0, tcpa=500.0, distance_ob=1000.0, v_rel=5.0))

        self.assertEqual(risk, 0.0)

    def test_negative_tcpa_does_not_increase_risk(self):
        risk = float(risk_calculations(dcpa=100.0, tcpa=-20.0, distance_ob=1000.0, v_rel=5.0))

        self.assertLess(risk, 0.4)


if __name__ == "__main__":
    unittest.main()
