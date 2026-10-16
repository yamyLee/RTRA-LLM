import unittest
import numpy as np
from src.navigation.velocity_obstacle import plan_velocity_obstacle
from src.dynamics.controller import controller


class VoSafetyTest(unittest.TestCase):
    def clearance(self, heading, position, target_speed=18.52, target_heading=np.pi, horizon=360):
        velocity = target_speed * np.array([np.cos(target_heading), np.sin(target_heading)])
        velocity -= 43.3 * np.array([np.cos(heading), np.sin(heading)])
        t = np.clip(-np.dot(position, velocity) / np.dot(velocity, velocity), 0, horizon)
        return np.linalg.norm(position + t * velocity)

    def test_right_turn_candidates_are_searched_for_safe_clearance(self):
        position = np.array([2000,-200])
        plan = plan_velocity_obstacle([2000/1852],[-200/1852],0,0,0,[18.52],[np.pi],kdir=1)
        self.assertLess(plan.correction, 0)
        self.assertGreater(self.clearance(plan.heading, position),600)
        self.assertTrue(plan.prediction_safe)
        self.assertAlmostEqual(plan.predicted_clearance_m, self.clearance(plan.heading, position))

    def test_absolute_evaluated_heading_is_preserved_when_guidance_differs(self):
        psi=.2
        plan=plan_velocity_obstacle([2000/1852],[-200/1852],0,0,psi,[18.52],[np.pi],desired_heading=-.1)
        self.assertAlmostEqual(plan.heading,psi+plan.correction)
        self.assertAlmostEqual(plan.predicted_clearance_m,self.clearance(plan.heading,np.array([2000,-200])))

    def test_hold_request_is_overridden_when_unsafe(self):
        plan=plan_velocity_obstacle([2000/1852],[-200/1852],0,0,0,[18.52],[np.pi],kdir=0)
        self.assertTrue(plan.direction_overridden)
        self.assertTrue(plan.prediction_safe)

    def test_initial_domain_intrusion_is_never_certified_safe(self):
        plan=plan_velocity_obstacle([100/1852],[0],0,0,0,[0],[0])
        self.assertFalse(plan.prediction_safe)
        self.assertEqual(plan.status,'no_safe_heading')

    def test_horizon_endpoint_intrusion_is_detected(self):
        plan=plan_velocity_obstacle([1000/1852],[0],0,0,0,[0],[0],horizon_s=10,kdir=0)
        self.assertTrue(plan.direction_overridden)
        self.assertTrue(plan.prediction_safe)

    def test_straight_course_with_no_targets(self):
        plan=plan_velocity_obstacle([],[],0,0,.2,[],[],desired_heading=.5)
        self.assertAlmostEqual(plan.heading,.5)
        self.assertTrue(plan.prediction_safe)

    def test_yaw_controller_takes_short_turn_across_pi(self):
        tau,_,_=controller(-np.pi+.01,np.pi-.01,0,10,0,0,.1)
        self.assertGreater(tau,0)
        self.assertLess(tau,3)
