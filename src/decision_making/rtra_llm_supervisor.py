"""Risk-triggered rule-aware LLM supervision for ship collision avoidance."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence

import numpy as np
from src.config.paper_parameters import LLM_FIXED_INTERVAL_STEPS, LLM_RISK_THRESHOLD

from src.decision_making.multi_llm_decision import COLREGSInterpreter, VesselState


class Action(str, Enum):
    """Discrete high-level maneuver actions used by the LLM supervisor."""

    STAND_ON = "Stand on, no action"
    STARBOARD = "Give-way, turn to starboard"
    PORT = "Give-way, turn to port"
    CONTINUE = "Continue current maneuver"


@dataclass
class ManeuverMemory:
    """Stateful maneuver context maintained across decision steps."""

    encounter_type: str = "unknown"
    last_candidate_action: Optional[Action] = None
    last_executed_action: Optional[Action] = None
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
            f"- Previous executed action: {last_action}\n"
            f"- Maneuver active: {active}\n"
            f"- Maneuver duration steps: {self.maneuver_steps}\n"
            "- If the previous maneuver remains compliant, prefer Continue current maneuver."
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


@dataclass(frozen=True)
class ValidationTarget:
    """High-risk target used for secondary COLREGs validation."""

    index: int
    encounter_type: str
    risk: float


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
        enable_memory: bool = True,
        enable_validator: bool = True,
        rule_only: bool = False,
    ):
        self.interpreter = COLREGSInterpreter(provider=provider)
        self.risk_threshold = float(risk_threshold)
        self.trigger_mode = trigger_mode
        self.fixed_interval = max(1, int(fixed_interval))
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

    @property
    def available(self) -> bool:
        """Whether the underlying LLM provider is configured and usable."""
        return self.interpreter.provider is not None

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
    ) -> DecisionResult:
        """Update the supervisor and return the current direction decision."""
        vessels = self._build_vessels(
            risk, distance, bearing, dcpa, tcpa, target_heading=target_heading
        )
        if not vessels:
            return DecisionResult(
                kdir=0,
                final_action=Action.STAND_ON,
                candidate_action=Action.STAND_ON,
                encounter_type="none",
                key_vessel_index=-1,
                key_risk=0.0,
                triggered=False,
                valid=True,
            )

        key_index = self._select_key_vessel(vessels)
        key_vessel = vessels[key_index]
        encounter_type = self._classify_encounter(
            key_vessel.bearing,
            target_heading=key_vessel.heading,
            own_heading=own_heading,
        )
        validation_targets = self._build_validation_targets(vessels, own_heading=own_heading)
        trigger_reasons = self._trigger_reasons(step, key_vessel.risk, encounter_type)
        triggered = bool(trigger_reasons)

        response = self.last_output
        candidate_action: Optional[Action] = None
        if triggered and self.rule_only:
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
            response = self.interpreter.make_decision(
                vessels,
                time_idx=step,
                memory_context=memory_context,
                encounter_type=encounter_type,
                key_vessel_index=key_index,
            )
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
                }
            )

        if candidate_action is None and not triggered and not response:
            candidate_action = (
                self.memory.last_executed_action
                if self.enable_memory and self.memory.last_executed_action is not None
                else self._default_safe_action(encounter_type)
            )
            valid = self._validate_action(candidate_action, encounter_type, key_vessel.risk, validation_targets)
            final_action = self._finalize_action(
                candidate_action,
                encounter_type,
                key_vessel.risk,
                valid,
                validation_targets,
            )
            final_valid = self._validate_action(final_action, encounter_type, key_vessel.risk, validation_targets)
            kdir = self._action_to_kdir(final_action)
            self._complete_trigger_event(
                step,
                triggered,
                candidate_action,
                final_action,
                final_valid,
                kdir,
                key_index,
                validation_targets,
            )
            self._update_memory(
                encounter_type=encounter_type,
                candidate_action=candidate_action,
                final_action=final_action,
                valid=final_valid,
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
        final_valid = self._validate_action(final_action, encounter_type, key_vessel.risk, validation_targets)
        kdir = self._action_to_kdir(final_action)
        self._complete_trigger_event(
            step,
            triggered,
            candidate_action,
            final_action,
            final_valid,
            kdir,
            key_index,
            validation_targets,
        )

        self._update_memory(
            encounter_type=encounter_type,
            candidate_action=candidate_action,
            final_action=final_action,
            valid=final_valid,
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
        )

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
        return [
            ValidationTarget(
                index=idx,
                encounter_type=self._classify_encounter(
                    vessel.bearing,
                    target_heading=vessel.heading,
                    own_heading=own_heading,
                ),
                risk=vessel.risk,
            )
            for idx, vessel in enumerate(vessels)
            if vessel.risk >= self.risk_threshold
        ]

    def _trigger_reasons(self, step: int, key_risk: float, encounter_type: str) -> List[str]:
        reasons: List[str] = []
        if self.trigger_mode == "always":
            return ["always"]

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
        if not self.memory.last_valid:
            reasons.append("previous_action_invalid")
        return reasons

    def _rule_based_action(
        self,
        key_vessel: VesselState,
        encounter_type: str,
        validation_targets: List[ValidationTarget],
    ) -> Action:
        """Select the deterministic action used by Rule-trigger baseline."""
        if encounter_type in {"head_on", "crossing_give_way"}:
            preferred = Action.STARBOARD
        elif encounter_type == "crossing_stand_on":
            preferred = Action.STAND_ON
        else:
            preferred = Action.STARBOARD if key_vessel.bearing >= 0 else Action.PORT

        if self._validate_action(
            preferred, encounter_type, key_vessel.risk, validation_targets
        ):
            return preferred
        return self._default_safe_action(encounter_type, validation_targets)

    def _classify_encounter(
        self,
        bearing: float,
        target_heading: Optional[float] = None,
        own_heading: Optional[float] = None,
    ) -> str:
        """Classify an encounter using bearing and, when available, heading.

        The paper's Case 7 describes TS1 as an overtaking vessel.  A bearing-only
        classifier labels a same-course target directly ahead as head-on, so the
        simulation supplies headings to disambiguate same-course overtaking from
        an opposing head-on encounter.  Calls that do not provide headings keep
        the historical bearing-only behavior used by the ablation tests.
        """
        bearing_deg = float(np.degrees(np.arctan2(np.sin(bearing), np.cos(bearing))))
        abs_bearing = abs(bearing_deg)

        if target_heading is not None and own_heading is not None:
            relative_heading = float(
                np.degrees(
                    np.arctan2(
                        np.sin(target_heading - own_heading),
                        np.cos(target_heading - own_heading),
                    )
                )
            )
            if abs_bearing <= 30 and abs(relative_heading) <= 30:
                return "overtaking"
            if abs_bearing <= 6 and abs(relative_heading) >= 150:
                return "head_on"

        if abs_bearing <= 6:
            return "head_on"
        if 6 < bearing_deg <= 112:
            return "crossing_give_way"
        if -118 <= bearing_deg < -6:
            return "crossing_stand_on"
        return "overtaking"

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
                # The simulation applies the newly returned direction on the
                # following control step.
                "effective_step": int(step + 1),
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

        if "continue current maneuver" in action_text or "continue current" in action_text:
            return Action.CONTINUE
        if "turn to starboard" in action_text or "turn starboard" in action_text or "starboard" in action_text:
            return Action.STARBOARD
        if "turn to port" in action_text or "turn port" in action_text:
            return Action.PORT
        if "stand on" in action_text or "no action" in action_text or "maintain course" in action_text:
            return Action.STAND_ON
        return None

    def _validate_action(
        self,
        action: Optional[Action],
        encounter_type: str,
        key_risk: float,
        validation_targets: Optional[List[ValidationTarget]] = None,
    ) -> bool:
        if not self.enable_validator:
            return action is not None
        if action is None:
            return False
        if action == Action.CONTINUE:
            if not self.enable_memory or self.memory.last_executed_action is None:
                return False
            return self._validate_action(
                self.memory.last_executed_action,
                encounter_type,
                key_risk,
                validation_targets,
            )

        targets = validation_targets
        if targets is None:
            targets = [ValidationTarget(index=-1, encounter_type=encounter_type, risk=key_risk)]
        if not targets:
            return action in {Action.STAND_ON, Action.STARBOARD, Action.PORT}
        return all(self._is_action_legal_for_target(action, target) for target in targets)

    def _is_action_legal_for_target(self, action: Action, target: ValidationTarget) -> bool:
        if target.risk < self.risk_threshold:
            return action in {Action.STAND_ON, Action.STARBOARD, Action.PORT}
        if target.encounter_type == "head_on":
            return action == Action.STARBOARD
        if target.encounter_type == "crossing_give_way":
            return action == Action.STARBOARD
        if target.encounter_type == "crossing_stand_on":
            return action in {Action.STAND_ON, Action.STARBOARD}
        if target.encounter_type == "overtaking":
            return action in {Action.STARBOARD, Action.PORT}
        return True

    def _finalize_action(
        self,
        candidate_action: Optional[Action],
        encounter_type: str,
        key_risk: float,
        valid: bool,
        validation_targets: Optional[List[ValidationTarget]] = None,
    ) -> Action:
        if valid and candidate_action == Action.CONTINUE:
            if self.enable_memory and self.memory.last_executed_action is not None:
                return self.memory.last_executed_action
            return self._default_safe_action(encounter_type, validation_targets)
        if valid and candidate_action is not None:
            return candidate_action
        if (
            self.enable_memory
            and self.memory.last_executed_action is not None
            and self._validate_action(self.memory.last_executed_action, encounter_type, key_risk, validation_targets)
        ):
            return self.memory.last_executed_action
        return self._default_safe_action(encounter_type, validation_targets)

    def _default_safe_action(
        self,
        encounter_type: str,
        validation_targets: Optional[List[ValidationTarget]] = None,
    ) -> Action:
        if validation_targets:
            for action in (Action.STARBOARD, Action.STAND_ON, Action.PORT):
                if all(self._is_action_legal_for_target(action, target) for target in validation_targets):
                    return action
        if encounter_type in {"head_on", "crossing_give_way", "overtaking"}:
            return Action.STARBOARD
        return Action.STAND_ON

    def _action_to_kdir(self, action: Action) -> int:
        if action == Action.STARBOARD:
            return 1
        if action == Action.PORT:
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
        if final_action == self.memory.last_executed_action and final_action != Action.STAND_ON:
            maneuver_steps = self.memory.maneuver_steps + 1
        else:
            maneuver_steps = 1 if final_action != Action.STAND_ON else 0

        self.memory = ManeuverMemory(
            encounter_type=encounter_type,
            last_candidate_action=candidate_action,
            last_executed_action=final_action,
            maneuver_active=final_action != Action.STAND_ON,
            maneuver_steps=maneuver_steps,
            last_valid=valid,
            last_response=response,
        )
