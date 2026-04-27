import unittest

from src.decision_making.rtra_llm_supervisor import Action, RiskTriggeredLLMSupervisor


class FakeProvider:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def generate_response(self, prompt):
        self.calls += 1
        return self.response


def make_supervisor(response, risk_threshold=0.30):
    supervisor = RiskTriggeredLLMSupervisor(provider="fake", risk_threshold=risk_threshold)
    fake_provider = FakeProvider(response)
    supervisor.interpreter.provider = fake_provider
    return supervisor, fake_provider


class RiskTriggeredLLMSupervisorTest(unittest.TestCase):
    def test_triggers_and_maps_starboard_action(self):
        supervisor, provider = make_supervisor(
            "Rule 14 (head-on), Action: Give-way, turn to starboard, Explanation: safe."
        )

        result = supervisor.update(
            risk=[0.3],
            distance=[1.0],
            bearing=[0.0],
            dcpa=[0.2],
            tcpa=[100.0],
            step=10,
        )

        self.assertTrue(result.triggered)
        self.assertTrue(result.valid)
        self.assertEqual(result.final_action, Action.STARBOARD)
        self.assertEqual(result.kdir, 1)
        self.assertEqual(provider.calls, 1)

    def test_low_risk_does_not_call_llm(self):
        supervisor, provider = make_supervisor(
            "Rule 14 (head-on), Action: Give-way, turn to starboard, Explanation: safe."
        )

        result = supervisor.update(
            risk=[0.1],
            distance=[3.0],
            bearing=[0.0],
            dcpa=[1.0],
            tcpa=[500.0],
            step=1,
        )

        self.assertFalse(result.triggered)
        self.assertEqual(provider.calls, 0)

    def test_invalid_head_on_port_turn_falls_back_to_safe_action(self):
        supervisor, _ = make_supervisor(
            "Rule 14 (head-on), Action: Give-way, turn to port, Explanation: unsafe."
        )

        result = supervisor.update(
            risk=[0.3],
            distance=[1.0],
            bearing=[0.0],
            dcpa=[0.2],
            tcpa=[100.0],
            step=10,
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.candidate_action, Action.PORT)
        self.assertEqual(result.final_action, Action.STARBOARD)
        self.assertEqual(result.kdir, 1)

    def test_invalid_action_does_not_cause_immediate_retry_loop(self):
        supervisor, provider = make_supervisor(
            "Rule 14 (head-on), Action: unclear, Explanation: malformed."
        )

        first = supervisor.update(
            risk=[0.3],
            distance=[1.0],
            bearing=[0.0],
            dcpa=[0.2],
            tcpa=[100.0],
            step=10,
        )
        second = supervisor.update(
            risk=[0.3],
            distance=[1.0],
            bearing=[0.0],
            dcpa=[0.2],
            tcpa=[100.0],
            step=11,
        )

        self.assertTrue(first.triggered)
        self.assertFalse(first.valid)
        self.assertFalse(second.triggered)
        self.assertEqual(provider.calls, 1)

    def test_invalid_memory_action_falls_back_for_new_encounter_type(self):
        supervisor, _ = make_supervisor(
            "Rule 13 (overtaking), Action: Stand on, no action, Explanation: unsafe."
        )
        supervisor.memory.last_executed_action = Action.STAND_ON

        result = supervisor.update(
            risk=[0.3],
            distance=[1.0],
            bearing=[3.14],
            dcpa=[0.2],
            tcpa=[100.0],
            step=10,
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.final_action, Action.STARBOARD)
        self.assertEqual(result.kdir, 1)

    def test_continue_is_invalid_when_memory_is_disabled(self):
        supervisor, _ = make_supervisor(
            "Rule 14 (head-on), Action: Continue current maneuver, Explanation: keep previous."
        )
        supervisor.enable_memory = False

        result = supervisor.update(
            risk=[0.3],
            distance=[1.0],
            bearing=[0.0],
            dcpa=[0.2],
            tcpa=[100.0],
            step=10,
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.final_action, Action.STARBOARD)


if __name__ == "__main__":
    unittest.main()
