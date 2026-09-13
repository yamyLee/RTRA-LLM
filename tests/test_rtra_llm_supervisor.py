import math
import unittest

from src.decision_making.rtra_llm_supervisor import Action, RiskTriggeredLLMSupervisor
from src.core.experiment_metrics import count_turn_events
from src.core.paper_experiment_utils import scene_level_values
from src.navigation.velocity_obstacle import apply_direction_constraint


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
    def test_paper_action_symbols_are_parsed(self):
        supervisor, _ = make_supervisor("Action: a_0")
        self.assertEqual(supervisor.parse_action("Action: a_0"), Action.A0)
        self.assertEqual(supervisor.parse_action("Action: a_R"), Action.AR)
        self.assertEqual(supervisor.parse_action("Action: a_L"), Action.AL)
        self.assertEqual(supervisor.parse_action("Action: a_C"), Action.AC)

    def test_direction_constraint_maps_actions_to_physical_turn_direction(self):
        self.assertLess(apply_direction_constraint(0.2, 1), 0.0)
        self.assertLess(apply_direction_constraint(-0.2, 1), 0.0)
        self.assertGreater(apply_direction_constraint(0.2, -1), 0.0)
        self.assertGreater(apply_direction_constraint(-0.2, -1), 0.0)
        self.assertLess(apply_direction_constraint(0.0, 1), 0.0)
        self.assertGreater(apply_direction_constraint(0.0, -1), 0.0)
        self.assertEqual(apply_direction_constraint(0.2, 0), 0.0)

    def test_without_risk_trigger_calls_at_226_high_level_updates(self):
        supervisor, provider = make_supervisor("Action: a_R")
        supervisor.trigger_mode = "high_level_always"
        supervisor.high_level_update_steps = 20
        for step in range(4500):
            supervisor.update([0.1], [1.0], [0.0], [0.2], [100.0], step=step)
        self.assertEqual(provider.calls, 226)
        self.assertEqual(supervisor.call_count, 226)

    def test_fixed_intervals_match_the_paper_call_counts(self):
        for interval, expected in ((100, 45), (250, 18), (500, 9), (750, 6)):
            with self.subTest(interval=interval):
                supervisor, provider = make_supervisor("Action: a_R")
                supervisor.trigger_mode = "fixed"
                supervisor.fixed_interval = interval
                for step in range(4500):
                    supervisor.update([0.1], [1.0], [0.0], [0.2], [100.0], step=step)
                self.assertEqual(provider.calls, expected)

    def test_experiment_statistics_average_seeds_before_scenes(self):
        rows = [
            {"case_number": 1, "seed": 42, "R_avg": 1.0},
            {"case_number": 1, "seed": 43, "R_avg": 3.0},
            {"case_number": 2, "seed": 42, "R_avg": 10.0},
            {"case_number": 2, "seed": 43, "R_avg": 14.0},
        ]
        self.assertEqual(scene_level_values(rows, "R_avg").tolist(), [2.0, 12.0])

    def test_turn_metric_counts_events_not_active_samples(self):
        self.assertEqual(count_turn_events([0, 1, 1, 1, 0, -1, -1, 0]), 2)

    def test_first_decision_is_an_initial_trigger(self):
        supervisor, provider = make_supervisor(
            "Rule 14 (head-on), Action: a_R, Explanation: safe."
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

    def test_paper_bearing_classifier_is_used_with_headings(self):
        supervisor, _ = make_supervisor("Action: a_R")
        result = supervisor.update(
            risk=[0.3],
            distance=[3.0],
            bearing=[0.0],
            dcpa=[1.0],
            tcpa=[500.0],
            step=1,
            target_heading=[0.0],
            own_heading=0.0,
        )
        self.assertEqual(result.encounter_type, "head_on")
        self.assertEqual(result.validation_encounter_types, ["head_on"])

    def test_paper_bearing_intervals_include_the_specified_boundaries(self):
        supervisor, _ = make_supervisor("Action: a_0")
        cases = [
            (-180, "overtaking"),
            (-118.001, "overtaking"),
            (-118, "crossing_stand_on"),
            (-118 - 1e-9, "overtaking"),
            (-118 + 1e-9, "crossing_stand_on"),
            (-6.001, "crossing_stand_on"),
            (-6, "head_on"),
            (-6 - 1e-9, "crossing_stand_on"),
            (0, "head_on"),
            (6, "head_on"),
            (6 + 1e-9, "crossing_give_way"),
            (6.001, "crossing_give_way"),
            (112, "crossing_give_way"),
            (112 + 1e-9, "overtaking"),
            (112.001, "overtaking"),
            (180, "overtaking"),
            (360, "head_on"),
            (390, "crossing_give_way"),
        ]
        for degrees, expected in cases:
            with self.subTest(degrees=degrees):
                self.assertEqual(supervisor._classify_encounter(math.radians(degrees)), expected)

    def test_secondary_target_invalidates_previous_action_before_model_call(self):
        supervisor, provider = make_supervisor("Action: a_0")
        state = dict(distance=[1.0, 1.1], bearing=[-0.5, 0.0],
                     dcpa=[0.2, 0.2], tcpa=[100.0, 100.0])
        first = supervisor.update(risk=[0.8, 0.29], step=1, **state)
        self.assertEqual(first.final_action, Action.A0)
        self.assertTrue(supervisor.memory.last_valid)

        provider.response = "Action: a_R"
        second = supervisor.update(risk=[0.8, 0.3], step=2, **state)
        self.assertEqual(second.key_vessel_index, first.key_vessel_index)
        self.assertEqual(second.encounter_type, first.encounter_type)
        self.assertEqual(second.key_risk, first.key_risk)
        self.assertEqual(second.trigger_reasons, ["previous_action_invalid"])
        self.assertEqual(second.final_action, Action.AR)
        self.assertEqual(provider.calls, 2)
        self.assertEqual(supervisor.trigger_history[-1]["reasons"], ["previous_action_invalid"])

        third = supervisor.update(risk=[0.8, 0.3], step=3, **state)
        self.assertFalse(third.triggered)
        self.assertEqual(provider.calls, 2)

    def test_secondary_encounter_change_triggers_without_risk_crossing(self):
        supervisor, provider = make_supervisor("Action: a_L")
        state = dict(risk=[0.8, 0.5], distance=[1.0, 1.1],
                     dcpa=[0.2, 0.2], tcpa=[100.0, 100.0])
        first = supervisor.update(bearing=[3.0, 3.0], step=1, **state)
        self.assertEqual(first.final_action, Action.AL)
        provider.response = "Action: a_C"
        second = supervisor.update(bearing=[3.0, 0.0], step=2, **state)
        self.assertEqual(second.trigger_reasons, ["previous_action_invalid"])
        self.assertEqual(provider.calls, 2)
        self.assertFalse(second.valid)
        self.assertEqual(second.final_action, Action.AR)

    def test_new_high_risk_target_does_not_trigger_if_previous_action_stays_legal(self):
        supervisor, provider = make_supervisor("Action: a_R")
        state = dict(distance=[1.0, 1.1], bearing=[-0.5, 0.0],
                     dcpa=[0.2, 0.2], tcpa=[100.0, 100.0])
        supervisor.update(risk=[0.8, 0.29], step=1, **state)
        second = supervisor.update(risk=[0.8, 0.3], step=2, **state)
        self.assertFalse(second.triggered)
        self.assertEqual(provider.calls, 1)

    def test_secondary_target_below_threshold_does_not_invalidate_previous_action(self):
        supervisor, provider = make_supervisor("Action: a_0")
        state = dict(distance=[1.0, 1.1], bearing=[-0.5, 0.0],
                     dcpa=[0.2, 0.2], tcpa=[100.0, 100.0])
        supervisor.update(risk=[0.8, 0.1], step=1, **state)
        second = supervisor.update(risk=[0.8, 0.29], step=2, **state)
        self.assertFalse(second.triggered)
        self.assertEqual(second.final_action, Action.A0)
        self.assertEqual(provider.calls, 1)

    def test_action_invalidation_preserves_fixed_and_always_call_schedules(self):
        for mode, expected_reasons, calls in [
            ("fixed", [], 1), ("always", ["always"], 2),
        ]:
            with self.subTest(mode=mode):
                supervisor, provider = make_supervisor("Action: a_0")
                supervisor.trigger_mode = mode
                supervisor.fixed_interval = 10
                state = dict(distance=[1.0, 1.1], bearing=[-0.5, 0.0],
                             dcpa=[0.2, 0.2], tcpa=[100.0, 100.0])
                supervisor.update(risk=[0.8, 0.29], step=1, **state)
                second = supervisor.update(risk=[0.8, 0.3], step=2, **state)
                self.assertEqual(second.trigger_reasons, expected_reasons)
                self.assertEqual(provider.calls, calls)
                self.assertEqual(second.final_action, Action.AR)

    def test_disabled_validator_does_not_add_rule_invalidation_trigger(self):
        supervisor, provider = make_supervisor("Action: a_0")
        supervisor.enable_validator = False
        state = dict(distance=[1.0, 1.1], bearing=[-0.5, 0.0],
                     dcpa=[0.2, 0.2], tcpa=[100.0, 100.0])
        supervisor.update(risk=[0.8, 0.29], step=1, **state)
        second = supervisor.update(risk=[0.8, 0.3], step=2, **state)
        self.assertFalse(second.triggered)
        self.assertEqual(second.final_action, Action.A0)
        self.assertEqual(provider.calls, 1)

    def test_no_targets_does_not_consume_initial_trigger(self):
        supervisor, provider = make_supervisor("Action: a_0")
        empty = supervisor.update([], [], [], [], [], step=0)
        self.assertFalse(empty.triggered)
        self.assertIsNone(supervisor.last_decision_step)
        first = supervisor.update([0.1], [1.0], [-0.5], [0.2], [100.0], step=1)
        self.assertEqual(first.trigger_reasons, ["initial_trigger"])
        self.assertEqual(provider.calls, 1)

    def test_equation_17_fallback_is_starboard_without_high_risk_targets(self):
        for response in ("Action: unclear", "Action: a_C"):
            with self.subTest(response=response):
                supervisor, _ = make_supervisor(response)
                result = supervisor.update([0.1], [1.0], [-0.5], [0.2], [100.0], step=1)
                self.assertEqual(result.encounter_type, "crossing_stand_on")
                self.assertEqual(result.high_risk_vessel_indices, [])
                self.assertFalse(result.valid)
                self.assertEqual(result.final_action, Action.AR)

    def test_legal_previous_action_takes_precedence_over_fallback(self):
        supervisor, provider = make_supervisor("Action: a_0")
        supervisor.update([0.1], [1.0], [-0.5], [0.2], [100.0], step=1)
        provider.response = "Action: unclear"
        result = supervisor.update([0.3], [1.0], [-0.5], [0.2], [100.0], step=2)
        self.assertFalse(result.valid)
        self.assertEqual(result.final_action, Action.A0)
        self.assertEqual(provider.calls, 2)

    def test_continue_resolves_to_the_legal_previous_action(self):
        supervisor, provider = make_supervisor("Action: a_0")
        supervisor.update([0.1], [1.0], [-0.5], [0.2], [100.0], step=1)
        provider.response = "Action: a_C"
        result = supervisor.update([0.3], [1.0], [-0.5], [0.2], [100.0], step=2)
        self.assertTrue(result.valid)
        self.assertEqual(result.candidate_action, Action.AC)
        self.assertEqual(result.final_action, Action.A0)
        self.assertEqual(result.kdir, 0)

    def test_invalid_previous_candidate_triggers_next_decision(self):
        supervisor, provider = make_supervisor("Action: a_R")
        state = dict(risk=[0.8], distance=[1.0], bearing=[0.0], dcpa=[0.2], tcpa=[100.0])
        supervisor.update(step=1, **state)
        supervisor.memory.last_valid = False
        result = supervisor.update(step=2, **state)
        self.assertTrue(result.triggered)
        self.assertEqual(provider.calls, 2)

    def test_rule_only_baseline_uses_the_same_action_invalidation_trigger(self):
        supervisor = RiskTriggeredLLMSupervisor(provider=None, rule_only=True)
        state = dict(distance=[1.0, 1.1], bearing=[-0.5, 0.0],
                     dcpa=[0.2, 0.2], tcpa=[100.0, 100.0])
        supervisor.update(risk=[0.8, 0.29], step=1, **state)
        second = supervisor.update(risk=[0.8, 0.3], step=2, **state)
        self.assertEqual(second.trigger_reasons, ["previous_action_invalid"])
        self.assertEqual(second.final_action, Action.AR)
        self.assertEqual(supervisor.call_count, 0)

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
        self.assertEqual(event["final_action"], Action.AR.value)
        self.assertEqual(event["kdir"], 1)
        self.assertEqual(event["effective_step"], 11)

    def test_encounter_type_change_triggers_below_risk_threshold(self):
        supervisor, provider = make_supervisor(
            "Rule 14 (head-on), Action: a_R, Explanation: safe."
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
        self.assertEqual(result.final_action, Action.AR)
        self.assertEqual(supervisor.call_count, 0)

    def test_triggers_and_maps_starboard_action(self):
        supervisor, provider = make_supervisor(
            "Rule 14 (head-on), Action: a_R, Explanation: safe."
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
        self.assertEqual(result.final_action, Action.AR)
        self.assertEqual(result.kdir, 1)
        self.assertEqual(provider.calls, 1)

    def test_low_risk_after_initial_decision_does_not_call_llm(self):
        supervisor, provider = make_supervisor(
            "Rule 14 (head-on), Action: a_R, Explanation: safe."
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
            "Rule 14 (head-on), Action: a_L, Explanation: unsafe."
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
        self.assertEqual(result.candidate_action, Action.AL)
        self.assertEqual(result.final_action, Action.AR)
        self.assertEqual(result.kdir, 1)

    def test_invalid_action_triggers_the_next_decision(self):
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
        self.assertTrue(second.triggered)
        self.assertEqual(provider.calls, 2)

    def test_invalid_memory_action_falls_back_for_new_encounter_type(self):
        supervisor, _ = make_supervisor(
            "Rule 13 (overtaking), Action: a_0, Explanation: unsafe."
        )
        supervisor.memory.last_executed_action = Action.A0

        result = supervisor.update(
            risk=[0.3],
            distance=[1.0],
            bearing=[3.14],
            dcpa=[0.2],
            tcpa=[100.0],
            step=10,
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.final_action, Action.AR)
        self.assertEqual(result.kdir, 1)

    def test_continue_is_invalid_when_memory_is_disabled(self):
        supervisor, _ = make_supervisor(
            "Rule 14 (head-on), Action: a_C, Explanation: keep previous."
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
        self.assertEqual(result.final_action, Action.AR)

    def test_multi_target_secondary_validation_blocks_action_invalid_for_other_high_risk_target(self):
        supervisor, _ = make_supervisor(
            "Rule 17 (stand-on), Action: a_0, Explanation: keep course."
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
        self.assertEqual(result.candidate_action, Action.A0)
        self.assertEqual(result.final_action, Action.AR)
        self.assertEqual(result.kdir, 1)
        self.assertEqual(result.high_risk_vessel_indices, [0, 1])
        self.assertEqual(result.validation_encounter_types, ["crossing_stand_on", "head_on"])


if __name__ == "__main__":
    unittest.main()
