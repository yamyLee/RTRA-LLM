import io
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from src.config.ppo_lstm import load_ppo_lstm_config
from src.core.ppo_lstm_environment import SawadaPpoLstmEnv
from src.core.simulation import run_simulation
from src.decision_making.rtra_llm_supervisor import RiskTriggeredLLMSupervisor, Action
from src.decision_making.multi_llm_decision import ChatOpenAIProvider
from src.navigation.velocity_obstacle import plan_velocity_obstacle
from src.dynamics.controller import controller
from src.dynamics.actuator_modeling import actuator_modeling
from src.dynamics.vessel_dynamics import vessel_dynamics


class FakeProvider:
    def generate_response(self, prompt):return 'Action: a_R'


class ExecutionAndLatencyTest(unittest.TestCase):
    def fake_run(self, **updates):
        supervisor=RiskTriggeredLLMSupervisor(provider='fake')
        supervisor.interpreter.provider=FakeProvider()
        args=SimpleNamespace(case_number=1,sim_time=.7,dt=.1,llm=1,save_runs=False,vo_prediction_model='constant')
        for key,value in updates.items():setattr(args,key,value)
        with patch('src.core.simulation.initialize_llm_supervisor',return_value=supervisor), \
             patch('src.core.simulation.get_obstacle_data',return_value=([20000.],[20000.],[18.52],[0.])), \
             redirect_stdout(io.StringIO()):
            result=run_simulation(args,True)
        return result,supervisor

    def test_generated_action_does_not_write_execution_memory(self):
        supervisor=RiskTriggeredLLMSupervisor(provider='fake')
        supervisor.interpreter.provider=FakeProvider()
        decision=supervisor.update([.1],[10],[0],[10],[500],step=0,target_heading=[0],own_heading=0)
        self.assertEqual(decision.final_action,Action.AR)
        self.assertIsNone(supervisor.memory.last_executed_action)
        supervisor.feedback_execution(0,0,0)
        self.assertEqual(supervisor.memory.last_executed_action,Action.A0)

    def test_ignored_low_risk_request_is_not_remembered_as_executed(self):
        trace,supervisor=self.fake_run()
        event=trace['llm_trigger_history'][0]
        self.assertEqual(event['final_action'],'a_R')
        self.assertEqual(event['executed_instruction'],'a_0')
        expected = Action.AR if trace['kdir'][-1] > 0 else Action.AL if trace['kdir'][-1] < 0 else Action.A0
        self.assertEqual(supervisor.memory.last_executed_action,expected)
        self.assertIsNone(supervisor.memory.active_maneuver_action)

    def test_observed_turn_is_distinct_from_applied_instruction(self):
        supervisor=RiskTriggeredLLMSupervisor(provider='fake')
        supervisor.feedback_execution(0,0,0,observed_heading_change=-.01)
        self.assertEqual(supervisor.memory.last_executed_action,Action.A0)
        self.assertEqual(supervisor.memory.observed_turn_action,Action.AR)

    def test_safe_hold_preserves_maneuver_without_retriggering(self):
        supervisor=RiskTriggeredLLMSupervisor(provider='fake',rule_only=True)
        observation=dict(risk=[.5],distance=[1.],bearing=[0.],dcpa=[.1],tcpa=[100.],
                         target_heading=[math.pi],own_heading=0.)
        supervisor.update(step=0,**observation)
        supervisor.feedback_execution(-.1,0,0,safe_prediction=True)
        supervisor.feedback_execution(0,0,1,safe_prediction=True)
        self.assertEqual(supervisor.memory.last_executed_action,Action.A0)
        self.assertEqual(supervisor.memory.active_maneuver_action,Action.AR)
        decision=supervisor.update(step=2,**observation)
        self.assertFalse(decision.triggered)
        self.assertTrue(decision.valid)

    def test_delayed_action_is_revalidated_on_arrival(self):
        from src.core.decision_delivery import DecisionDelivery
        from src.core.simulation import ensure_arg_defaults
        args=ensure_arg_defaults(SimpleNamespace(seed=42,llm_latency_mode='fixed',llm_latency_s=.3))
        delivery=DecisionDelivery(args,.1)
        supervisor=RiskTriggeredLLMSupervisor(provider='fake')
        class LeftProvider:
            def generate_response(self,prompt):return 'Action: a_L'
        supervisor.interpreter.provider=LeftProvider()
        initial=dict(risk=[.1],distance=[1.],bearing=[0.],dcpa=[.1],tcpa=[100.],target_heading=[0.],own_heading=0.)
        self.assertIsNone(delivery.advance(0,supervisor,initial))
        current={**initial,'risk':[.8],'target_heading':[math.pi]}
        decision=delivery.advance(3,supervisor,current)
        self.assertEqual(decision.final_action,Action.AR)
        self.assertFalse(decision.valid)
        self.assertEqual(supervisor.trigger_history[0]['delivery_status'],'accepted_fallback')

    def test_persistent_planner_override_is_a_single_invalidation_event(self):
        supervisor=RiskTriggeredLLMSupervisor(provider='fake',rule_only=True)
        observation=dict(risk=[.5],distance=[1.],bearing=[-math.pi/4],dcpa=[.1],tcpa=[100.],
                         target_heading=[-math.pi/2],own_heading=0.)
        supervisor.update(step=0,**observation)
        for step in range(1,10):
            supervisor.feedback_execution(.1,0,step-1,safe_prediction=True)
            decision=supervisor.update(step=step,**observation)
        self.assertFalse(decision.triggered)
        self.assertEqual(len(supervisor.trigger_history),2)
        self.assertTrue(decision.valid)
        changed={**observation,'tcpa':[20.]}
        decision=supervisor.update(step=10,**changed)
        self.assertIn('rule_constraints_changed',decision.trigger_reasons)

    def test_continue_without_active_encounter_does_not_lock_route_heading(self):
        supervisor=RiskTriggeredLLMSupervisor(provider='fake')
        supervisor.memory.last_executed_action=Action.A0
        supervisor.memory.last_applied_heading=.4
        supervisor.memory.active_maneuver_action=None
        class ContinueProvider:
            def generate_response(self,prompt):return 'Action: a_C'
        supervisor.interpreter.provider=ContinueProvider()
        with patch('src.core.simulation.initialize_llm_supervisor',return_value=supervisor), \
             patch('src.core.simulation.get_obstacle_data',return_value=([20000.],[20000.],[18.52],[0.])), \
             redirect_stdout(io.StringIO()):
            trace=run_simulation(SimpleNamespace(case_number=1,sim_time=.1,dt=.1,llm=1,
                save_runs=False,vo_prediction_model='constant'),True)
        self.assertAlmostEqual(trace['desired_heading_rad'][0],0)

    def test_fixed_delay_advances_motion_and_delivers_on_simulated_tick(self):
        trace,_=self.fake_run(llm_latency_mode='fixed',llm_latency_s=.3)
        event=trace['llm_trigger_history'][0]
        self.assertEqual(event['effective_step'],3)
        self.assertAlmostEqual(trace['x'][3],43.3*.3)
        self.assertEqual(event['delivery_status'],'accepted')

    def test_timeout_response_is_never_applied(self):
        trace,_=self.fake_run(llm_latency_mode='fixed',llm_latency_s=2,llm_timeout_s=.2)
        event=trace['llm_trigger_history'][0]
        self.assertEqual(event['delivery_status'],'discarded_timeout')
        self.assertIsNone(event['effective_step'])
        self.assertEqual(trace['llm_discarded_decisions'],1)
        self.assertAlmostEqual(trace['x'][3],43.3*.3)

    def test_stale_response_is_never_applied(self):
        trace,_=self.fake_run(llm_latency_mode='fixed',llm_latency_s=.3,llm_max_decision_age_s=.1)
        self.assertEqual(trace['llm_trigger_history'][0]['delivery_status'],'discarded_stale')
        self.assertIsNone(trace['llm_trigger_history'][0]['effective_step'])

    def test_pending_response_at_end_is_reported(self):
        trace,_=self.fake_run(sim_time=.2,llm_latency_mode='fixed',llm_latency_s=1)
        self.assertTrue(trace['llm_pending_at_end'])
        self.assertEqual(trace['llm_trigger_history'][0]['delivery_status'],'pending')
        self.assertIsNone(trace['llm_trigger_history'][0]['effective_step'])

    def test_measured_delay_uses_wall_latency_without_sleeping(self):
        with patch('src.decision_making.rtra_llm_supervisor.time.perf_counter',side_effect=[0.,.35]):
            trace,_=self.fake_run(llm_latency_mode='measured')
        self.assertEqual(trace['llm_trigger_history'][0]['effective_step'],4)

    def test_random_delay_is_reproducible(self):
        first,_=self.fake_run(llm_latency_mode='random',llm_latency_s=.3,llm_latency_jitter_s=.1,seed=42)
        second,_=self.fake_run(llm_latency_mode='random',llm_latency_s=.3,llm_latency_jitter_s=.1,seed=42)
        self.assertEqual(first['llm_trigger_history'][0]['simulated_latency_s'],second['llm_trigger_history'][0]['simulated_latency_s'])
        np.testing.assert_allclose(first['psi'],second['psi'])

    def test_injected_failure_holds_previous_command(self):
        trace,_=self.fake_run(llm_failure_probability=1.)
        self.assertEqual(trace['llm_trigger_history'][0]['delivery_status'],'discarded_injected_failure')
        self.assertIsNone(trace['llm_trigger_history'][0]['effective_step'])
        self.assertEqual(trace['llm_discarded_decisions'],1)

    def test_provider_failure_clears_usage_and_is_not_parse_failure(self):
        provider=ChatOpenAIProvider.__new__(ChatOpenAIProvider)
        provider.provider_name='fake'
        class Client:
            def invoke(self,prompt):
                if prompt=='bad':raise RuntimeError('offline test error')
                return SimpleNamespace(content='Action: a_R',usage_metadata={'total_tokens':17})
        provider.client=Client()
        provider.generate_response('good')
        self.assertEqual(provider.last_usage_metadata['total_tokens'],17)
        provider.generate_response('bad')
        self.assertEqual(provider.last_usage_metadata,{})
        self.assertFalse(provider.last_call_ok)
        supervisor=RiskTriggeredLLMSupervisor(provider='fake')
        class ErrorProvider:
            last_call_ok=False
            last_call_error='test'
            def generate_response(self,prompt):return 'test error'
        supervisor.interpreter.provider=ErrorProvider()
        supervisor.update([.1],[10],[0],[10],[500],step=0,target_heading=[0],own_heading=0)
        self.assertEqual(supervisor.call_failure_count,1)
        self.assertEqual(supervisor.received_response_count,0)
        self.assertEqual(supervisor.parse_failure_count,0)

    def test_time_limit_is_not_waypoint_success(self):
        config=load_ppo_lstm_config();config={**config,'simulation':dict(config['simulation'])}
        config['simulation']['episode_time_s']=10
        env=SawadaPpoLstmEnv(config,training=False);env.reset(case_number=1,seed=42)
        _,_,_,info=env.step(0)
        self.assertFalse(info['success'])
        self.assertFalse(env.summary()['success'])
        self.assertTrue(env.summary()['collision_free_episode'])

    def test_ppo_training_collision_uses_hulls_not_center_radius(self):
        config=load_ppo_lstm_config();config={**config,'simulation':dict(config['simulation'])}
        config['simulation']['episode_time_s']=10
        with patch('src.core.ppo_lstm_environment.get_obstacle_data',return_value=([0.],[54.],[18.52],[0.])):
            env=SawadaPpoLstmEnv(config,training=True);env.reset(case_number=1,seed=42)
            env.step(0)
        self.assertFalse(env.summary()['collision'])

    def test_nomoto_command_survives_fine_step_execution(self):
        state=np.zeros(6);target=np.array([2000.,-200.]);velocity=np.array([-18.52,0.])
        plan=plan_velocity_obstacle([target[0]/1852],[target[1]/1852],0,0,0,[18.52],[math.pi],
                                   kdir=1,dynamics_state=state)
        self.assertEqual(plan.prediction_model,'nomoto_closed_loop')
        self.assertTrue(plan.prediction_safe)
        integral=0.;minimum=float('inf')
        with patch('src.dynamics.vessel_dynamics.np.random.randn',return_value=0.):
            for _ in range(3600):
                minimum=min(minimum,float(np.linalg.norm(target-state[:2])))
                torque,speed,integral=controller(plan.heading,state[2],state[3],43.3,state[4],integral,.1)
                state+=.1*vessel_dynamics(state,[actuator_modeling(torque),speed])
                target+=.1*velocity
        self.assertGreater(minimum,600)


if __name__=='__main__':unittest.main()
