import unittest

from src.decision_making.rtra_llm_supervisor import Action, RiskTriggeredLLMSupervisor
from src.core.experiment_metrics import count_turn_events


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
    def test_turn_metric_counts_events_not_active_samples(self):
        self.assertEqual(count_turn_events([0, 1, 1, 1, 0, -1, -1, 0]), 2)

    def test_first_decision_is_an_initial_trigger(self):
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

        self.assertTrue(result.triggered)
        self.assertIn("initial_trigger", result.trigger_reasons)
        self.assertEqual(provider.calls, 1)

    def test_same_course_target_ahead_is_overtaking_when_headings_are_known(self):
        supervisor, _ = make_supervisor("Action: Give-way, turn to starboard")
        result = supervisor.update(
            risk=[0.1],
            distance=[3.0],
            bearing=[0.0],
            dcpa=[1.0],
            tcpa=[500.0],
            step=1,
            target_heading=[0.0],
            own_heading=0.0,
        )
        self.assertEqual(result.encounter_type, "overtaking")

    def test_trigger_history_records_action_and_effective_step(self):
        supervisor = RiskTriggeredLLMSupervisor(provider=None, rule_only=True)
        supervisor.update(
            risk=[0.3],
            distance=[1.0],
            bearing=[0.0],
            dcpa=[0.2],
            tcpa=[100.0],
            step=10,
        )
        event = supervisor.trigger_history[0]
        self.assertEqual(event["final_action"], Action.STARBOARD.value)
        self.assertEqual(event["kdir"], 1)
        self.assertEqual(event["effective_step"], 11)

    def test_encounter_type_change_triggers_below_risk_threshold(self):
        supervisor, provider = make_supervisor(
            "Rule 14 (head-on), Action: Give-way, turn to starboard, Explanation: safe."
        )
        supervisor.update([0.3], [1.0], [0.0], [0.2], [100.0], step=1)

        result = supervisor.update(
            risk=[0.1],
            distance=[3.0],
            bearing=[1.0],
            dcpa=[1.0],
            tcpa=[500.0],
            step=2,
        )

        self.assertTrue(result.triggered)
        self.assertIn("encounter_type_change", result.trigger_reasons)
        self.assertEqual(provider.calls, 2)

    def test_rule_only_baseline_has_zero_llm_calls(self):
        supervisor = RiskTriggeredLLMSupervisor(
            provider=None,
            rule_only=True,
            risk_threshold=0.30,
        )

        result = supervisor.update(
            risk=[0.3],
            distance=[1.0],
            bearing=[0.0],
            dcpa=[0.2],
            tcpa=[100.0],
            step=1,
        )

        self.assertTrue(result.triggered)
        self.assertEqual(result.final_action, Action.STARBOARD)
        self.assertEqual(supervisor.call_count, 0)

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

    def test_low_risk_after_initial_decision_does_not_call_llm(self):
        supervisor, provider = make_supervisor(
            "Rule 14 (head-on), Action: Give-way, turn to starboard, Explanation: safe."
        )

        supervisor.update(
            risk=[0.1],
            distance=[3.0],
            bearing=[0.0],
            dcpa=[1.0],
            tcpa=[500.0],
            step=0,
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
        self.assertEqual(provider.calls, 1)

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

    def test_multi_target_secondary_validation_blocks_action_invalid_for_other_high_risk_target(self):
        supervisor, _ = make_supervisor(
            "Rule 17 (stand-on), Action: Stand on, no action, Explanation: keep course."
        )

        result = supervisor.update(
            risk=[0.35, 0.31],
            distance=[1.0, 1.1],
            bearing=[-0.5, 0.0],
            dcpa=[0.2, 0.2],
            tcpa=[100.0, 100.0],
            step=10,
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.candidate_action, Action.STAND_ON)
        self.assertEqual(result.final_action, Action.STARBOARD)
        self.assertEqual(result.kdir, 1)
        self.assertEqual(result.high_risk_vessel_indices, [0, 1])
        self.assertEqual(result.validation_encounter_types, ["crossing_stand_on", "head_on"])


if __name__ == "__main__":
    unittest.main()
