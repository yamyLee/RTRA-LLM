"""Risk-triggered rule-aware LLM supervision for ship collision avoidance."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence

import numpy as np
import re
import time
from src.navigation.encounters import classify_encounter
from src.config.paper_parameters import OWN_SHIP_SPEED_MPS, TARGET_SHIP_SPEED_MPS, VO_SAFETY_RADIUS_M
from src.config.paper_parameters import (
    LLM_FIXED_INTERVAL_STEPS,
    LLM_HIGH_LEVEL_UPDATE_STEPS,
    LLM_RISK_THRESHOLD,
)

from src.decision_making.multi_llm_decision import COLREGSInterpreter, VesselState


class Action(str, Enum):
    A0 = "a_0"
    AR = "a_R"
    AL = "a_L"
    AC = "a_C"


@dataclass
class ManeuverMemory:
    """Stateful maneuver context maintained across decision steps."""

    encounter_type: str = "unknown"
    last_candidate_action: Optional[Action] = None
    last_executed_action: Optional[Action] = None
    last_planned_action: Optional[Action] = None
    active_maneuver_action: Optional[Action] = None
    last_applied_heading: Optional[float] = None
    observed_turn_action: Optional[Action] = None
    maneuver_active: bool = False
    maneuver_steps: int = 0
    last_valid: bool = True
    last_response: str = ""

    def as_prompt_context(self) -> str:
        """Return a compact text representation for the LLM prompt."""
        last_action = self.last_executed_action.value if self.last_executed_action else "None"
        active = "true" if self.maneuver_active else "false"
        return (
            "Decision memory:\n"
            f"- Previous encounter type: {self.encounter_type}\n"
            f"- Previous applied instruction: {last_action}\n"
            f"- Observed turn: {self.observed_turn_action.value if self.observed_turn_action else 'None'}\n"
            f"- Maneuver active: {active}\n"
            f"- Active maneuver: {self.active_maneuver_action.value if self.active_maneuver_action else 'None'}\n"
            f"- Maneuver duration steps: {self.maneuver_steps}\n"
            "- If the previous maneuver remains compliant, prefer a_C."
        )


@dataclass
class DecisionResult:
    """Result returned by one supervisor update."""

    kdir: int
    final_action: Action
    candidate_action: Optional[Action]
    encounter_type: str
    key_vessel_index: int
    key_risk: float
    triggered: bool
    valid: bool
    response: str = ""
    trigger_reasons: List[str] = field(default_factory=list)
    high_risk_vessel_indices: List[int] = field(default_factory=list)
    validation_encounter_types: List[str] = field(default_factory=list)
    rule_conflict: bool = False


@dataclass(frozen=True)
class ValidationTarget:
    """High-risk target used for secondary COLREGs validation."""

    index: int
    encounter_type: str
    risk: float
    emergency: bool = False


class RiskTriggeredLLMSupervisor:
    """High-level RTRA-LLM supervisor.

    The supervisor implements the paper method:
    risk-triggered sparse LLM calls, maneuver memory, structured action parsing,
    COLREGs-oriented validation, and mapping to the low-level direction factor.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        risk_threshold: float = LLM_RISK_THRESHOLD,
        trigger_mode: str = "risk",
        fixed_interval: int = LLM_FIXED_INTERVAL_STEPS,
        high_level_update_steps: int = LLM_HIGH_LEVEL_UPDATE_STEPS,
        enable_memory: bool = True,
        enable_validator: bool = True,
        rule_only: bool = False,
    ):
        self.interpreter = COLREGSInterpreter(provider="offline-rules" if rule_only else provider)
        self.risk_threshold = float(risk_threshold)
        self.trigger_mode = trigger_mode
        self.fixed_interval = max(1, int(fixed_interval))
        self.high_level_update_steps = max(1, int(high_level_update_steps))
        self.enable_memory = enable_memory
        self.enable_validator = enable_validator
        self.rule_only = rule_only

        self.memory = ManeuverMemory()
        self.last_output = ""
        self.last_key_risk = 0.0
        self.last_encounter_type = "unknown"
        self.last_decision_step: Optional[int] = None
        self.last_call_step: Optional[int] = None
        self.call_count = 0
        self.trigger_history: List[dict] = []
        self._encounter_roles = {}
        self._current_encounters = []
        self._last_validation_targets = []
        self.rule_conflict = False
        self._previous_rule_conflict = False
        self.parse_failure_count = 0
        self.call_failure_count = 0
        self.received_response_count = 0
        self._execution_invalid_active = False
        self._last_constraint_signature = None

    @property
    def available(self) -> bool:
        """Whether the underlying LLM provider is configured and usable."""
        return self.interpreter.provider is not None

    @property
    def model_name(self) -> str:
        provider = self.interpreter.provider
        provider = getattr(provider, "provider", provider)
        return str(getattr(provider, "model", ""))

    def update(
        self,
        risk: Sequence[float],
        distance: Sequence[float],
        bearing: Sequence[float],
        dcpa: Sequence[float],
        tcpa: Sequence[float],
        step: int,
        target_heading: Optional[Sequence[float]] = None,
        own_heading: Optional[float] = None,
        own_speed: float = OWN_SHIP_SPEED_MPS,
        target_speed: Optional[Sequence[float]] = None,
    ) -> DecisionResult:
        """Update the supervisor and return the current direction decision."""
        observation = self.observe(risk, distance, bearing, dcpa, tcpa, step,
                                   target_heading, own_heading, own_speed, target_speed)
        if observation is None:
            return DecisionResult(
                kdir=0,
                final_action=Action.A0,
                candidate_action=Action.A0,
                encounter_type="none",
                key_vessel_index=-1,
                key_risk=0.0,
                triggered=False,
                valid=True,
            )
        vessels, key_index, encounter_type, validation_targets = observation
        key_vessel = vessels[key_index]
        trigger_reasons = self._trigger_reasons(
            step, key_vessel.risk, encounter_type, validation_targets
        )
        triggered = bool(trigger_reasons)

        response = self.last_output
        candidate_action: Optional[Action] = None
        if triggered and self.rule_only:
            self.last_call_step = step
            candidate_action = self._rule_based_action(
                key_vessel, encounter_type, validation_targets
            )
            response = ""
            self.trigger_history.append(
                {
                    "step": step,
                    "risk": key_vessel.risk,
                    "encounter_type": encounter_type,
                    "reasons": trigger_reasons,
                    "llm_called": False,
                }
            )
        elif triggered and self.available:
            memory_context = self.memory.as_prompt_context() if self.enable_memory else None
            call_started = time.perf_counter()
            response = self.interpreter.make_decision(
                vessels,
                time_idx=step,
                memory_context=memory_context,
                encounter_type=encounter_type,
                key_vessel_index=key_index,
                own_heading=own_heading, own_speed=own_speed,
            )
            call_latency = time.perf_counter() - call_started
            provider = getattr(self.interpreter.provider, "provider", self.interpreter.provider)
            call_ok = getattr(provider, "last_call_ok", True)
            self.call_failure_count += int(not call_ok)
            self.received_response_count += int(call_ok)
            self.parse_failure_count += int(call_ok and self.parse_action(response) is None)
            self.last_output = response
            self.last_call_step = step
            self.call_count += 1
            self.trigger_history.append(
                {
                    "step": step,
                    "risk": key_vessel.risk,
                    "encounter_type": encounter_type,
                    "reasons": trigger_reasons,
                    "llm_called": True,
                    "response": response,
                    "call_ok": bool(call_ok),
                    "call_error": getattr(provider, "last_call_error", None),
                    "prompt": getattr(self.interpreter, "last_prompt", ""),
                    "inference_latency_s": call_latency,
                    "usage": getattr(getattr(self.interpreter.provider, "provider", self.interpreter.provider),
                                     "last_usage_metadata", {}),
                }
            )

        if candidate_action is None and not triggered and not response:
            candidate_action = (
                (self.memory.last_planned_action or self.memory.active_maneuver_action or self.memory.last_executed_action)
                if self.enable_memory and self.memory.last_planned_action is not None
                else self._default_safe_action(validation_targets)
            )
            valid = self._validate_action(candidate_action, encounter_type, key_vessel.risk, validation_targets)
            final_action = self._finalize_action(
                candidate_action,
                encounter_type,
                key_vessel.risk,
                valid,
                validation_targets,
            )
            kdir = self._action_to_kdir(final_action)
            self._complete_trigger_event(
                step,
                triggered,
                candidate_action,
                final_action,
                valid,
                kdir,
                key_index,
                validation_targets,
            )
            self._update_memory(
                encounter_type=encounter_type,
                candidate_action=candidate_action,
                final_action=final_action,
                valid=valid,
                response=response,
            )
            self.last_key_risk = key_vessel.risk
            self.last_encounter_type = encounter_type
            self.last_decision_step = step
            return DecisionResult(
                kdir=kdir,
                final_action=final_action,
                candidate_action=candidate_action,
                encounter_type=encounter_type,
                key_vessel_index=key_index,
                key_risk=key_vessel.risk,
                triggered=False,
                valid=valid,
                response=response,
                trigger_reasons=trigger_reasons,
                high_risk_vessel_indices=[target.index for target in validation_targets],
                validation_encounter_types=[target.encounter_type for target in validation_targets],
                rule_conflict=self.rule_conflict,
            )

        if candidate_action is None:
            candidate_action = self.parse_action(response)
        valid = self._validate_action(candidate_action, encounter_type, key_vessel.risk, validation_targets)
        final_action = self._finalize_action(
            candidate_action,
            encounter_type,
            key_vessel.risk,
            valid,
            validation_targets,
        )
        kdir = self._action_to_kdir(final_action)
        self._complete_trigger_event(
            step,
            triggered,
            candidate_action,
            final_action,
            valid,
            kdir,
            key_index,
            validation_targets,
        )

        self._update_memory(
            encounter_type=encounter_type,
            candidate_action=candidate_action,
            final_action=final_action,
            valid=valid,
            response=response,
        )
        self.last_key_risk = key_vessel.risk
        self.last_encounter_type = encounter_type
        self.last_decision_step = step

        return DecisionResult(
            kdir=kdir,
            final_action=final_action,
            candidate_action=candidate_action,
            encounter_type=encounter_type,
            key_vessel_index=key_index,
            key_risk=key_vessel.risk,
            triggered=triggered,
            valid=valid,
            response=response,
            trigger_reasons=trigger_reasons,
            high_risk_vessel_indices=[target.index for target in validation_targets],
            validation_encounter_types=[target.encounter_type for target in validation_targets],
            rule_conflict=self.rule_conflict,
        )

    def observe(self, risk, distance, bearing, dcpa, tcpa, step,
                target_heading=None, own_heading=None, own_speed=OWN_SHIP_SPEED_MPS,
                target_speed=None):
        """Refresh geometry and constraints without generating a new decision."""
        vessels = self._build_vessels(
            risk, distance, bearing, dcpa, tcpa, target_heading=target_heading
        )
        if not vessels:
            self._last_validation_targets = []
            self._encounter_roles.clear()
            self._current_observation = None
            return None

        key_index = self._select_key_vessel(vessels)
        key_vessel = vessels[key_index]
        speeds = ([TARGET_SHIP_SPEED_MPS] * len(vessels)
                  if target_speed is None else list(target_speed))
        if len(speeds) != len(vessels):
            raise ValueError("Target speed length differs from target count")
        self._current_encounters = []
        for idx, (vessel, speed) in enumerate(zip(vessels, speeds)):
            vessel.speed = float(speed)
            geometric = self._classify_encounter(vessel.bearing, vessel.heading,
                                                  own_heading, own_speed, speed)
            # Retain responsibility throughout the maneuver until past CPA and
            # outside the planning safety domain. This state is independent of
            # optional maneuver memory, including in the no-memory ablation.
            clear = (vessel.tcpa < 0 and vessel.distance * 1852 > VO_SAFETY_RADIUS_M)
            if clear:
                self._encounter_roles.pop(idx, None)
            elif idx not in self._encounter_roles and vessel.risk >= self.risk_threshold:
                if geometric not in {"unknown", "none"}:
                    self._encounter_roles[idx] = geometric
            self._current_encounters.append(self._encounter_roles.get(idx, geometric))
        encounter_type = self._current_encounters[key_index]
        validation_targets = self._build_validation_targets(vessels, own_heading=own_heading)
        self._last_validation_targets = validation_targets
        self._previous_rule_conflict = self.rule_conflict
        self.rule_conflict = bool(validation_targets) and not any(
            all(self._is_action_legal_for_target(action, target) for target in validation_targets)
            for action in (Action.A0, Action.AR, Action.AL)
        )
        self._current_observation = (vessels, key_index, encounter_type, validation_targets)
        return self._current_observation

    def _build_vessels(
        self,
        risk: Sequence[float],
        distance: Sequence[float],
        bearing: Sequence[float],
        dcpa: Sequence[float],
        tcpa: Sequence[float],
        target_heading: Optional[Sequence[float]] = None,
    ) -> List[VesselState]:
        risk_arr = np.atleast_1d(risk)
        distance_arr = np.atleast_1d(distance)
        bearing_arr = np.atleast_1d(bearing)
        dcpa_arr = np.atleast_1d(dcpa)
        tcpa_arr = np.atleast_1d(tcpa)
        heading_arr = (
            np.atleast_1d(target_heading)
            if target_heading is not None
            else np.full(risk_arr.shape, np.nan)
        )

        arrays = (risk_arr, distance_arr, bearing_arr, dcpa_arr, tcpa_arr, heading_arr)
        if any(arr.ndim != 1 or arr.size != risk_arr.size for arr in arrays):
            raise ValueError("Encounter arrays must be one-dimensional with equal lengths")
        if not all(np.all(np.isfinite(arr)) for arr in (risk_arr, distance_arr, bearing_arr, dcpa_arr)):
            raise ValueError("Risk, range, bearing and DCPA must be finite")
        if np.any(distance_arr < 0) or np.any(np.isnan(tcpa_arr)):
            raise ValueError("Range must be nonnegative and TCPA cannot be NaN")

        return [
            VesselState(
                float(r),
                float(d),
                float(b),
                float(dc),
                float(tc),
                None if np.isnan(h) else float(h),
            )
            for r, d, b, dc, tc, h in zip(
                risk_arr, distance_arr, bearing_arr, dcpa_arr, tcpa_arr, heading_arr
            )
        ]

    def _select_key_vessel(self, vessels: List[VesselState]) -> int:
        if not vessels:
            return 0
        return int(np.argmax([v.risk for v in vessels]))

    def _build_validation_targets(
        self, vessels: List[VesselState], own_heading: Optional[float] = None
    ) -> List[ValidationTarget]:
        """Build the high-risk target set for secondary rule validation."""
        targets = []
        for idx, vessel in enumerate(vessels):
            if vessel.risk < self.risk_threshold and idx not in self._encounter_roles:
                continue
            encounter = (self._current_encounters[idx] if idx < len(self._current_encounters)
                         else self._classify_encounter(vessel.bearing, vessel.heading, own_heading))
            if encounter in {"none", "unknown"} or (vessel.tcpa < 0 and vessel.distance * 1852 > VO_SAFETY_RADIUS_M):
                continue
            # Explicit emergency proxy, NOT a full Rule 17 non-compliance detector.
            emergency = (vessel.distance * 1852 <= VO_SAFETY_RADIUS_M or
                         (0 <= vessel.tcpa <= 30 and abs(vessel.dcpa) * 1852 <= VO_SAFETY_RADIUS_M))
            targets.append(ValidationTarget(idx, encounter, max(vessel.risk, self.risk_threshold), emergency))
        return targets

    def _trigger_reasons(
        self,
        step: int,
        key_risk: float,
        encounter_type: str,
        validation_targets: List[ValidationTarget],
    ) -> List[str]:
        reasons: List[str] = []
        if self.trigger_mode == "always":
            return ["always"]

        if self.trigger_mode == "high_level_always":
            if self.last_decision_step is None or (step + 1) % self.high_level_update_steps == 0:
                return ["high_level_update"]
            return []

        if self.trigger_mode == "fixed":
            if self.last_call_step is None or step - self.last_call_step >= self.fixed_interval:
                return ["fixed_interval"]
            return []

        if self.last_decision_step is None:
            reasons.append("initial_trigger")
        if key_risk >= self.risk_threshold and self.last_key_risk < self.risk_threshold:
            reasons.append("risk_threshold_crossing")
        if (
            encounter_type != self.last_encounter_type
            and self.last_decision_step is not None
        ):
            reasons.append("encounter_type_change")
        signature = tuple((target.index,target.encounter_type,target.emergency) for target in validation_targets)
        if (self.last_decision_step is not None and self._last_constraint_signature is not None
                and signature != self._last_constraint_signature):
            reasons.append("rule_constraints_changed")
        self._last_constraint_signature = signature
        if self.last_decision_step is not None:
            persistent_conflict = self.enable_validator and self.rule_conflict and self._previous_rule_conflict
            if not self.memory.last_valid and not persistent_conflict:
                reasons.append("previous_candidate_invalid")
            execution_invalid = not self._validate_action(
                self.memory.active_maneuver_action or self.memory.last_executed_action,
                encounter_type,key_risk,validation_targets)
            # Invalid execution is an event on entry, not a demand to invoke the
            # model on every tick while the planner keeps overriding the same
            # physically infeasible instruction. Constraint changes still trigger.
            if execution_invalid and not self._execution_invalid_active and not persistent_conflict:
                reasons.append("previous_execution_invalid")
            self._execution_invalid_active = execution_invalid
        return reasons

    def _rule_based_action(
        self,
        key_vessel: VesselState,
        encounter_type: str,
        validation_targets: List[ValidationTarget],
    ) -> Action:
        """Select the deterministic action used by Rule-trigger baseline."""
        if encounter_type in {"head_on", "crossing_give_way"}:
            preferred = Action.AR
        elif encounter_type in {"crossing_stand_on", "being_overtaken"}:
            preferred = Action.A0
        else:
            preferred = Action.AR if key_vessel.bearing >= 0 else Action.AL

        if self._validate_action(
            preferred, encounter_type, key_vessel.risk, validation_targets
        ):
            return preferred
        return self._default_safe_action(validation_targets)

    def _classify_encounter(self, bearing, target_heading=None, own_heading=None,
                            own_speed=OWN_SHIP_SPEED_MPS, target_speed=TARGET_SHIP_SPEED_MPS):
        return classify_encounter(bearing, target_heading, own_heading, own_speed, target_speed)

    def _complete_trigger_event(
        self,
        step: int,
        triggered: bool,
        candidate_action: Optional[Action],
        final_action: Action,
        valid: bool,
        kdir: int,
        key_index: int,
        validation_targets: List[ValidationTarget],
    ) -> None:
        """Attach the executed decision to the corresponding trigger record."""
        if not triggered or not self.trigger_history:
            return
        event = self.trigger_history[-1]
        if event.get("step") != step:
            return
        event.update(
            {
                "candidate_action": candidate_action.value if candidate_action else None,
                "final_action": final_action.value,
                "valid": bool(valid),
                "kdir": int(kdir),
                "key_vessel_index": int(key_index),
                "high_risk_vessel_indices": [target.index for target in validation_targets],
                "validation_encounter_types": [
                    target.encounter_type for target in validation_targets
                ],
                # Decision precedes integration on the same synchronous state.
                "effective_step": int(step),
                "rule_conflict": self.rule_conflict,
            }
        )

    def parse_action(self, response: str) -> Optional[Action]:
        """Parse the structured Action field from an LLM response."""
        if not response:
            return None
        text = response.lower()

        action_text = text
        marker = "action:"
        if marker in text:
            action_text = text.split(marker, 1)[1]
            if "explanation:" in action_text:
                action_text = action_text.split("explanation:", 1)[0]

        symbols = re.findall(r"(?<![a-z0-9_])a_?[rlc0](?![a-z0-9_])", action_text)
        mapping = {"ar": Action.AR, "al": Action.AL, "ac": Action.AC, "a0": Action.A0}
        actions = {mapping[token.replace("_", "")] for token in symbols}
        return actions.pop() if len(actions) == 1 else None

    def _validate_action(
        self,
        action: Optional[Action],
        encounter_type: str,
        key_risk: float,
        validation_targets: Optional[List[ValidationTarget]] = None,
    ) -> bool:
        if action is None:
            return False
        if action == Action.AC:
            if not self.enable_memory or self.memory.last_executed_action is None:
                return False
            return self._validate_action(
                self.memory.active_maneuver_action or self.memory.last_executed_action,
                encounter_type,
                key_risk,
                validation_targets,
            )

        if not self.enable_validator:
            return True
        targets = validation_targets
        if targets is None:
            targets = [ValidationTarget(index=-1, encounter_type=encounter_type, risk=key_risk)]
        if not targets:
            return action in {Action.A0, Action.AR, Action.AL}
        return all(self._is_action_legal_for_target(action, target) for target in targets)

    def _is_action_legal_for_target(self, action: Action, target: ValidationTarget) -> bool:
        if target.risk < self.risk_threshold:
            return action in {Action.A0, Action.AR, Action.AL}
        if target.encounter_type == "head_on":
            return action == Action.AR
        if target.encounter_type == "crossing_give_way":
            return action == Action.AR
        if target.encounter_type in {"crossing_stand_on", "being_overtaken"}:
            return action in ({Action.A0, Action.AR} if target.emergency else {Action.A0})
        if target.encounter_type == "overtaking":
            return action in {Action.AR, Action.AL}
        return True

    def _finalize_action(
        self,
        candidate_action: Optional[Action],
        encounter_type: str,
        key_risk: float,
        valid: bool,
        validation_targets: Optional[List[ValidationTarget]] = None,
    ) -> Action:
        if valid and candidate_action == Action.AC:
            if self.enable_memory and self.memory.last_executed_action is not None:
                return self.memory.active_maneuver_action or self.memory.last_executed_action
            return self._default_safe_action(validation_targets)
        if valid and candidate_action is not None:
            return candidate_action
        if (
            self.enable_memory
            and self.memory.last_executed_action is not None
            and self._validate_action(self.memory.active_maneuver_action or self.memory.last_executed_action,
                                      encounter_type, key_risk, validation_targets)
        ):
            return self.memory.active_maneuver_action or self.memory.last_executed_action
        return self._default_safe_action(validation_targets)

    def _default_safe_action(
        self,
        validation_targets: Optional[List[ValidationTarget]] = None,
    ) -> Action:
        targets = validation_targets or []
        for action in (Action.AR, Action.A0, Action.AL):
            if all(self._is_action_legal_for_target(action, target) for target in targets):
                return action
        # No common rule-legal action: explicitly flagged as a conflict above.
        # The simulator delegates to unconstrained VO and records the outcome.
        return Action.A0

    def _action_to_kdir(self, action: Action) -> int:
        if action == Action.AR:
            return 1
        if action == Action.AL:
            return -1
        return 0

    def _update_memory(
        self,
        encounter_type: str,
        candidate_action: Optional[Action],
        final_action: Action,
        valid: bool,
        response: str,
    ) -> None:
        # Generation is not execution. Feedback from the planner/integrator
        # exclusively updates applied instructions and observed motion.
        self.memory.encounter_type = encounter_type
        self.memory.last_candidate_action = candidate_action
        self.memory.last_planned_action = final_action
        self.memory.last_valid = valid
        self.memory.last_response = response

    def feedback_execution(self, heading, own_heading, step, observed_heading_change=0.0, safe_prediction=None):
        from src.navigation.encounters import wrap_angle
        offset = float(wrap_angle(heading - own_heading))
        applied = Action.AR if offset < -1e-10 else Action.AL if offset > 1e-10 else Action.A0
        previous_maneuver = self.memory.active_maneuver_action
        active = applied if applied != Action.A0 and self._last_validation_targets else None
        observation = getattr(self, '_current_observation', None)
        safe_hold = bool(observation) and all(
            vessel.tcpa < 0 or abs(vessel.dcpa)*1852 > VO_SAFETY_RADIUS_M
            for idx, vessel in enumerate(observation[0])
            if idx in {target.index for target in self._last_validation_targets})
        if safe_prediction is not None:
            safe_hold = bool(safe_prediction)
        if applied == Action.A0 and previous_maneuver is not None and self._last_validation_targets and safe_hold:
            active = previous_maneuver
        self.memory.maneuver_steps = (self.memory.maneuver_steps + 1
            if active is not None and active == previous_maneuver else int(active is not None))
        self.memory.last_executed_action = applied
        self.memory.active_maneuver_action = active
        self.memory.last_applied_heading = float(heading)
        self.memory.maneuver_active = active is not None
        self.memory.observed_turn_action = (Action.AR if observed_heading_change < -1e-10 else
            Action.AL if observed_heading_change > 1e-10 else Action.A0)
        return applied

    def revalidate(self, decision):
        """Recheck a delayed candidate against the latest observed constraints."""
        from dataclasses import replace
        observation = getattr(self, '_current_observation', None)
        if observation is None:
            return replace(decision, final_action=Action.A0, kdir=0, valid=False,
                           high_risk_vessel_indices=[], validation_encounter_types=[])
        vessels, key_index, encounter, targets = observation
        candidate = decision.candidate_action
        valid = self._validate_action(candidate, encounter, vessels[key_index].risk, targets)
        final = self._finalize_action(candidate, encounter, vessels[key_index].risk, valid, targets)
        return replace(decision, final_action=final, kdir=self._action_to_kdir(final),
            valid=valid, encounter_type=encounter, key_vessel_index=key_index,
            key_risk=vessels[key_index].risk, rule_conflict=self.rule_conflict,
            high_risk_vessel_indices=[t.index for t in targets],
            validation_encounter_types=[t.encounter_type for t in targets])
