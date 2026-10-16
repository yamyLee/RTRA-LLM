import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from src.core.experiment_metrics import trajectory_metrics, rectangular_hull_overlap
from src.core.run_archive import save_run
from src.core.simulation import run_simulation
from src.config.ppo_lstm import load_ppo_lstm_config
from src.core.ppo_lstm_environment import SawadaPpoLstmEnv


def simple_trace(positions, target=None):
    pos=np.asarray(positions,float)
    n=len(pos)-1
    if target is None:
        target=np.broadcast_to([5000.,5000.], (len(pos),1,2)).copy()
    target=np.asarray(target,float)
    return dict(time=np.arange(n,dtype=float), x=pos[:-1,0],y=pos[:-1,1],psi=np.zeros(n),
        kdir=np.zeros(n),risk=np.zeros((n,1)),dcpa=np.full((n,1),900.),tcpa=np.ones((n,1)),
        dt=1.,final_position_m=pos[-1],final_heading_rad=0.,
        obstacles_x_history=target[:-1,:,0],obstacles_y_history=target[:-1,:,1],
        final_target_positions_m=target[-1],obstacles_heading_rad=np.zeros(1),
        waypoints_x=np.array([0,1]),waypoints_y=np.array([0,0]))


class CommonMetricsTest(unittest.TestCase):
    def test_paper_distance_distinguishes_equal_length_paths(self):
        reference=simple_trace([[0,0],[1,0],[2,0],[3,0]])
        detour=simple_trace([[0,0],[1,0],[1,1],[2,1]])
        value=trajectory_metrics(detour,reference)
        self.assertAlmostEqual(value['path_length_m'],3)
        self.assertAlmostEqual(value['delta_D_pct'],-29.289)
        self.assertAlmostEqual(value['final_dist_nm'],np.sqrt(2)/1852)
        self.assertAlmostEqual(value['baseline_dist_nm'],2/1852)
        self.assertAlmostEqual(value['route_progress_m'],2)
        self.assertAlmostEqual(value['max_cross_track_m'],1)

    def test_paper_distance_does_not_charge_extra_length_with_same_endpoint(self):
        reference=simple_trace([[0,0],[1,0],[2,0],[3,0]])
        detour=simple_trace([[0,0],[.5,1],[2,0],[3,0]])
        value=trajectory_metrics(detour,reference)
        self.assertGreater(value['path_length_m'],trajectory_metrics(reference)['path_length_m'])
        self.assertEqual(value['delta_D_pct'],0)

    def test_paper_distance_uses_last_recorded_sample_before_final_state(self):
        reference=simple_trace([[0,0],[1,0],[2,0]])
        trace=simple_trace([[0,0],[0,2],[100,0]])
        value=trajectory_metrics(trace,reference)
        self.assertEqual(value['delta_D_pct'],100)
        self.assertAlmostEqual(value['final_dist_nm'],2/1852)

    def test_paper_distance_retains_zero_reference_convention(self):
        reference=simple_trace([[0,0],[0,0],[1,0]])
        trace=simple_trace([[0,0],[1,1],[2,0]])
        value=trajectory_metrics(trace,reference)
        self.assertEqual(value['baseline_dist_nm'],0)
        self.assertEqual(value['delta_D_pct'],0)

    def test_predicted_dcpa_and_actual_range_are_separate(self):
        trace=simple_trace([[0,0],[1,0],[2,0]])
        trace['dcpa'][:]=0
        self.assertEqual(trajectory_metrics(trace)['min_dcpa_nm'],0)
        self.assertGreater(trajectory_metrics(trace)['min_actual_separation_nm'],3)

    def test_dcpa_statistic_excludes_past_cpa(self):
        trace=simple_trace([[0,0],[1,0],[2,0]])
        trace['dcpa'][:,0]=[0,900]
        trace['tcpa'][:,0]=[-1,10]
        self.assertAlmostEqual(trajectory_metrics(trace)['min_dcpa_nm'],900/1852)

    def test_domain_intrusion_count_and_duration_include_endpoint(self):
        trace=simple_trace([[0,0],[0,0],[0,0]],target=[[[700,0]],[[500,0]],[[500,0]]])
        metrics=trajectory_metrics(trace)
        self.assertEqual(metrics['safety_domain_intrusion_count'],1)
        self.assertEqual(metrics['safety_domain_intrusion_duration_s'],1)
        self.assertFalse(metrics['collision'])

    def test_collision_uses_rectangles_not_planning_radius(self):
        p=np.array([[0,0]])
        self.assertTrue(rectangular_hull_overlap(p,[0],np.array([[[50,0]]]),[0])[0,0])
        self.assertFalse(rectangular_hull_overlap(p,[0],np.array([[[60,0]]]),[0])[0,0])
        self.assertFalse(rectangular_hull_overlap(p,[0],np.array([[[0,30]]]),[0])[0,0])

    def test_paired_metrics_reject_different_time_grids(self):
        trace=simple_trace([[0,0],[1,0],[2,0]])
        other={**trace,'time':np.array([.1,1.1])}
        with self.assertRaises(ValueError):trajectory_metrics(trace,other)

    def test_archive_keeps_numeric_arrays_and_metadata(self):
        trace=simple_trace([[0,0],[1,0],[2,0]])
        trace['metrics']=trajectory_metrics(trace)
        trace['llm_trigger_history']=[{'response':'Action: a_R','prompt':'test situation'}]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(save_run(trace,SimpleNamespace(output_dir=folder,case_number=1,seed=42)))
            data=np.load(path/'trace.npz',allow_pickle=False)
            self.assertEqual(data['x'].tolist(),[0,1])
            meta=json.loads((path/'run.json').read_text())
            self.assertIn('source_sha256',meta['source'])
            self.assertEqual(meta['llm_trigger_history'][0]['response'],'Action: a_R')
            self.assertNotIn('environment',meta)
            self.assertTrue((path/'source_snapshot.zip').is_file())
            from src.core.run_archive import load_run
            loaded=load_run(path)
            self.assertAlmostEqual(trajectory_metrics(loaded)['path_length_m'],2)

    def test_own_and_target_samples_share_time_zero(self):
        from src.utils.imazu_cases import get_obstacle_data
        with redirect_stdout(io.StringIO()):
            trace=run_simulation(SimpleNamespace(case_number=7,sim_time=1.,dt=.1,save_runs=False),True)
        xo,yo,speeds,headings=get_obstacle_data(7)
        np.testing.assert_allclose(trace['obstacles_x_history'][0],xo)
        np.testing.assert_allclose(trace['obstacles_y_history'][0],yo)
        np.testing.assert_allclose(trace['distance_ob'][0],np.hypot(xo,yo))
        self.assertEqual(trace['x'][0],0)
        self.assertEqual(trace['time'][0],0)

    def test_main_applies_the_absolute_heading_evaluated_by_vo(self):
        from src.navigation.velocity_obstacle import plan_velocity_obstacle
        plans=[]
        def capture(*args,**kwargs):
            plan=plan_velocity_obstacle(*args,**kwargs);plans.append(plan);return plan
        with patch('src.core.simulation.plan_velocity_obstacle',side_effect=capture):
            with redirect_stdout(io.StringIO()):
                trace=run_simulation(SimpleNamespace(case_number=7,sim_time=1.,dt=.1,save_runs=False),True)
        np.testing.assert_allclose(trace['desired_heading_rad'],[p.heading for p in plans])

    def test_ppo_dense_trace_uses_common_metric_formulas(self):
        config=load_ppo_lstm_config()
        config={**config,'simulation':dict(config['simulation'])}
        config['simulation']['episode_time_s']=10.
        env=SawadaPpoLstmEnv(config,training=False)
        env.reset(case_number=7,seed=42)
        env.step(0)
        trace=env.evaluation_trajectory()
        self.assertEqual(len(trace['time']),100)
        self.assertEqual(trace['time'][0],0)
        metrics=trajectory_metrics(trace,trace)
        self.assertEqual(metrics['A_turn_pct'],100)
        self.assertAlmostEqual(metrics['delta_D_pct'],0)
        self.assertAlmostEqual(metrics['path_length_m'],43.3*10)
        self.assertTrue(np.isnan(env.summary()['A_turn_pct']))

    def test_paper_runners_export_identical_metric_definitions(self):
        import importlib
        import os
        reference=simple_trace([[0,0],[1,0],[2,0],[3,0]])
        trace=simple_trace([[0,0],[1,0],[1,1],[2,1]])
        expected=trajectory_metrics(trace,reference)
        condition=dict(id=1,name='test',label='test',paper_row='test',llm=1,
                       disable_memory=False,disable_rule_validator=False,file='test',trigger_mode='risk')
        with patch.dict(os.environ,{'ALL_PROXY':''}):
            multi=importlib.import_module('run_multi_model_comparison')
            module=importlib.import_module('run_module_ablation')
            prompt=importlib.import_module('run_prompt_ablation')
            fixed=importlib.import_module('run_fixed_interval_sensitivity')
            baseline=importlib.import_module('run_llm_baseline')
            threshold=importlib.import_module('run_threshold_sweep')
        results=[multi._extract_metrics(trace,reference,1,'qwen','qwen'),
                 module._extract_metrics(trace,reference,1,condition,'qwen'),
                 prompt._extract_metrics(trace,reference,1,condition,'qwen'),
                 baseline._extract_metrics(trace,reference,1,'qwen',250),
                 threshold._extract_metrics(trace,reference,1,.3,'qwen'),
                 fixed._extract_metrics(trace,reference,case_number=1,provider='qwen',method='test',
                    schedule='risk',interval_steps=None,configuration='matched',sim_time=3,dt=1)]
        for result in results:
            self.assertAlmostEqual(result['delta_D_pct'],expected['delta_D_pct'])
            self.assertAlmostEqual(result['final_dist_nm'],expected['final_dist_nm'])
            self.assertAlmostEqual(result['baseline_dist_nm'],expected['baseline_dist_nm'])
            self.assertEqual(result['A_turn_pct'],expected['A_turn_pct'])
            self.assertEqual(result['metric_schema'],expected['metric_schema'])

    def test_ppo_evaluation_runner_pairs_and_archives_without_trained_network(self):
        # Exercise the evaluation/export plumbing with a deterministic stub, not
        # a claimed trained PPO policy. No PyTorch or model download is needed.
        from src.core.ppo_lstm_simulation import _run_one
        config=load_ppo_lstm_config()
        config={**config,'simulation':dict(config['simulation'])}
        config['simulation']['episode_time_s']=10.
        class StubAgent:
            def __init__(self):self.config=config
            def initial_state(self):return None
            def act(self,*args,**kwargs):return 0.,None,None,None,None
        with tempfile.TemporaryDirectory() as folder:
            checkpoint=Path(folder)/'stub.pt';checkpoint.write_bytes(b'test only')
            args=SimpleNamespace(case_number=7,seed=42,output_dir=folder)
            with patch('src.core.ppo_lstm_simulation.require_torch'):
                with redirect_stdout(io.StringIO()):
                    result=_run_one(args,config,checkpoint,Path(folder)/'evaluation',
                                    agent=StubAgent(),checkpoint_metadata={})
            self.assertTrue(np.isfinite(result['A_turn_pct']))
            self.assertAlmostEqual(result['path_length_m'],433.)
            self.assertTrue((Path(result['archive_path'])/'trace.npz').is_file())
            self.assertIn('checkpoint_sha256',result)


if __name__=='__main__':unittest.main()
