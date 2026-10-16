import math
import unittest
from src.decision_making.rtra_llm_supervisor import Action, RiskTriggeredLLMSupervisor
from src.navigation.encounters import classify_encounter
from src.core.experiment_metrics import count_turn_events, count_reverse_switches
from src.core.paper_experiment_utils import scene_level_values


class FakeProvider:
    def __init__(self, response):
        self.response, self.calls = response, 0

    def generate_response(self, prompt):
        self.calls += 1
        return self.response


def make_supervisor(response='Action: a_R', **kwargs):
    supervisor = RiskTriggeredLLMSupervisor(provider='fake', **kwargs)
    provider = FakeProvider(response)
    supervisor.interpreter.provider = provider
    return supervisor, provider


def state(**updates):
    value = dict(risk=[0.5], distance=[1.0], bearing=[0.0], dcpa=[0.1], tcpa=[100.0],
                 target_heading=[math.pi], own_heading=0.0)
    value.update(updates)
    return value


class RiskTriggeredLLMSupervisorTest(unittest.TestCase):
    def test_forward_slow_target_is_overtaking_not_head_on(self):
        self.assertEqual(classify_encounter(0, 0, 0, 43.3, 18.52), 'overtaking')
        self.assertEqual(classify_encounter(0, math.pi, 0, 43.3, 18.52), 'head_on')

    def test_target_approaching_from_astern_is_being_overtaken(self):
        self.assertEqual(classify_encounter(math.pi, 0, 0, 10, 20), 'being_overtaken')
        self.assertEqual(classify_encounter(math.pi, 0, 0, 20, 10), 'none')

    def test_missing_headings_do_not_claim_a_colregs_encounter(self):
        self.assertEqual(classify_encounter(0, None, 0, 10, 10), 'unknown')

    def test_crossing_responsibility_uses_starboard_bearing(self):
        self.assertEqual(classify_encounter(math.pi / 4, math.pi / 2, 0, 20, 20), 'crossing_give_way')
        self.assertEqual(classify_encounter(-math.pi / 4, -math.pi / 2, 0, 20, 20), 'crossing_stand_on')

    def test_symmetric_geometry_after_heading_rotation_has_same_role(self):
        for rotation in (0, 1.2, -2, 6.28):
            self.assertEqual(classify_encounter(0, rotation, rotation, 20, 10), 'overtaking')

    def test_ambiguous_output_is_rejected(self):
        supervisor, _ = make_supervisor()
        for symbol, expected in [('a_R', Action.AR), ('a_L', Action.AL), ('a_C', Action.AC), ('a_0', Action.A0)]:
            self.assertEqual(supervisor.parse_action('Action: ' + symbol), expected)
        self.assertIsNone(supervisor.parse_action('Action: a_R or a_L'))
        self.assertIsNone(supervisor.parse_action('Action: unclear'))
        self.assertEqual(supervisor.parse_action('Action: a_R, Explanation: do not use a_L'), Action.AR)

    def test_head_on_port_is_rejected_and_fallback_is_right(self):
        supervisor, _ = make_supervisor('Action: a_L')
        result = supervisor.update(step=0, **state())
        self.assertFalse(result.valid)
        self.assertEqual(result.final_action, Action.AR)

    def test_stand_on_ship_holds_course_outside_emergency_proxy(self):
        supervisor, _ = make_supervisor('Action: a_R')
        result = supervisor.update(step=0, **state(bearing=[-math.pi / 4], target_heading=[-math.pi / 2]))
        self.assertEqual(result.encounter_type, 'crossing_stand_on')
        self.assertFalse(result.valid)
        self.assertEqual(result.final_action, Action.A0)

    def test_stand_on_ship_can_take_right_action_in_explicit_emergency(self):
        supervisor, _ = make_supervisor('Action: a_R')
        result = supervisor.update(step=0, **state(bearing=[-math.pi / 4], target_heading=[-math.pi / 2], tcpa=[20]))
        self.assertTrue(result.valid)
        self.assertEqual(result.final_action, Action.AR)

    def test_conflicting_multi_target_rules_are_explicit(self):
        supervisor, _ = make_supervisor()
        result = supervisor.update(step=0, risk=[.5, .5], distance=[1, 1], bearing=[0, -math.pi/4],
            dcpa=[.1, .1], tcpa=[100, 100], target_heading=[math.pi, -math.pi/2], own_heading=0)
        self.assertTrue(result.rule_conflict)
        self.assertFalse(result.valid)
        self.assertEqual(supervisor.trigger_history[-1]['rule_conflict'], True)

    def test_overtaking_responsibility_persists_during_maneuver(self):
        supervisor, _ = make_supervisor()
        supervisor.update(step=0, **state(target_heading=[0]))
        second = supervisor.update(step=1, **state(bearing=[.6], target_heading=[0], own_heading=-.6))
        self.assertEqual(second.encounter_type, 'overtaking')
        passed = supervisor.update(step=2, **state(bearing=[math.pi], target_heading=[0], tcpa=[-20], risk=[.1]))
        self.assertNotIn(0, supervisor._encounter_roles)
        self.assertEqual(passed.high_risk_vessel_indices, [])

    def test_continue_uses_last_executed_maneuver(self):
        supervisor, provider = make_supervisor(trigger_mode='fixed', fixed_interval=1)
        supervisor.update(step=0, **state())
        supervisor.feedback_execution(-.1,0,0)
        provider.response = 'Action: a_C'
        result = supervisor.update(step=1, **state())
        self.assertEqual(result.candidate_action, Action.AC)
        self.assertEqual(result.final_action, Action.AR)
        self.assertTrue(result.valid)

    def test_continue_is_rejected_without_memory(self):
        supervisor, _ = make_supervisor('Action: a_C', enable_memory=False)
        self.assertFalse(supervisor.update(step=0, **state()).valid)

    def test_parse_failure_is_counted_per_received_call_not_per_timestep(self):
        supervisor, provider = make_supervisor('Action: unclear', trigger_mode='fixed', fixed_interval=10)
        for step in range(5):
            supervisor.update(step=step, **state())
        self.assertEqual(provider.calls, 1)
        self.assertEqual(supervisor.parse_failure_count, 1)
        self.assertIn('prompt', supervisor.trigger_history[0])

    def test_trigger_records_candidate_and_execution_invalidity_separately(self):
        supervisor, provider = make_supervisor('Action: unclear')
        supervisor.update(step=0, **state())
        supervisor.feedback_execution(-.1,0,0)
        result = supervisor.update(step=1, **state())
        self.assertIn('previous_candidate_invalid', result.trigger_reasons)
        self.assertNotIn('previous_execution_invalid', result.trigger_reasons)

    def test_secondary_target_invalidates_hold_without_changing_key_target(self):
        supervisor, provider = make_supervisor('Action: a_0')
        inputs = dict(distance=[1, 1], bearing=[-math.pi/4, 0], dcpa=[.1, .1], tcpa=[100,100],
                      target_heading=[-math.pi/2, math.pi], own_heading=0)
        supervisor.update(step=0, risk=[.8,.2], **inputs)
        second = supervisor.update(step=1, risk=[.8,.4], **inputs)
        self.assertIn('previous_execution_invalid', second.trigger_reasons)
        self.assertTrue(second.rule_conflict)
        self.assertEqual(provider.calls, 2)

    def test_unchanged_rule_conflict_does_not_reinvoke_every_tick(self):
        supervisor, provider = make_supervisor()
        inputs = dict(risk=[.8,.4],distance=[1,1],bearing=[0,-math.pi/4],dcpa=[.1,.1],
            tcpa=[100,100],target_heading=[math.pi,-math.pi/2],own_heading=0)
        for step in range(10):
            result=supervisor.update(step=step,**inputs)
        self.assertTrue(result.rule_conflict)
        self.assertEqual(provider.calls,1)

    def test_encounter_array_length_mismatch_is_rejected(self):
        supervisor, _ = make_supervisor()
        with self.assertRaises(ValueError):
            supervisor.update(step=0,**state(target_heading=[]))

    def test_fixed_schedules_keep_original_call_counts(self):
        for interval, count in ((100,45),(250,18),(500,9),(750,6)):
            supervisor, provider = make_supervisor(trigger_mode='fixed', fixed_interval=interval)
            for step in range(4500):
                supervisor.update(step=step, **state(risk=[.1]))
            self.assertEqual(provider.calls, count)

    def test_no_trigger_ablation_has_226_updates(self):
        supervisor, provider = make_supervisor(trigger_mode='high_level_always', high_level_update_steps=20)
        for step in range(4500):
            supervisor.update(step=step, **state(risk=[.1]))
        self.assertEqual(provider.calls, 226)

    def test_decision_is_effective_before_same_step_integration(self):
        supervisor, _ = make_supervisor()
        supervisor.update(step=10, **state())
        self.assertEqual(supervisor.trigger_history[0]['effective_step'], 10)

    def test_rule_baseline_calls_no_llm(self):
        supervisor = RiskTriggeredLLMSupervisor(provider='fake', rule_only=True)
        result = supervisor.update(step=0, **state())
        self.assertEqual(result.final_action, Action.AR)
        self.assertEqual(supervisor.call_count, 0)

    def test_turn_metrics_distinguish_events_and_reversals(self):
        self.assertEqual(count_turn_events([0,1,1,0,-1,-1,0,1]), 3)
        self.assertEqual(count_reverse_switches([0,1,1,0,-1,-1,0,1]), 2)

    def test_average_seeds_before_scenes(self):
        rows = [{'case_number':1,'R_avg':1}, {'case_number':1,'R_avg':3}, {'case_number':2,'R_avg':10}]
        self.assertEqual(scene_level_values(rows,'R_avg').tolist(), [2,10])


if __name__ == '__main__':
    unittest.main()
