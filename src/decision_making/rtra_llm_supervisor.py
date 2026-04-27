"""Risk-triggered rule-aware LLM supervision for ship collision avoidance."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence

import numpy as np

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


class RiskTriggeredLLMSupervisor:
    """High-level RTRA-LLM supervisor.

    The supervisor implements the paper method:
    risk-triggered sparse LLM calls, maneuver memory, structured action parsing,
    COLREGs-oriented validation, and mapping to the low-level direction factor.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        risk_threshold: float = 0.30,
        trigger_mode: str = "risk",
        fixed_interval: int = 200,
        enable_memory: bool = True,
        enable_validator: bool = True,
    ):
        self.interpreter = COLREGSInterpreter(provider=provider)
        self.risk_threshold = float(risk_threshold)
        self.trigger_mode = trigger_mode
        self.fixed_interval = max(1, int(fixed_interval))
        self.enable_memory = enable_memory
        self.enable_validator = enable_validator

        self.memory = ManeuverMemory()
        self.last_output = ""
        self.last_key_risk = 0.0
        self.last_encounter_type = "unknown"
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
    ) -> DecisionResult:
        """Update the supervisor and return the current direction decision."""
        vessels = self._build_vessels(risk, distance, bearing, dcpa, tcpa)
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
        encounter_type = self._classify_encounter(key_vessel.bearing)
        trigger_reasons = self._trigger_reasons(step, key_vessel.risk, encounter_type)
        triggered = bool(trigger_reasons)

        response = self.last_output
        if triggered and self.available:
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
                }
            )

        if not triggered and not response:
            candidate_action = (
                self.memory.last_executed_action
                if self.enable_memory and self.memory.last_executed_action is not None
                else self._default_safe_action(encounter_type)
            )
            valid = True
            final_action = candidate_action
            kdir = self._action_to_kdir(final_action)
            self._update_memory(
                encounter_type=encounter_type,
                candidate_action=candidate_action,
                final_action=final_action,
                valid=valid,
                response=response,
            )
            self.last_key_risk = key_vessel.risk
            self.last_encounter_type = encounter_type
            return DecisionResult(
                kdir=kdir,
                final_action=final_action,
                candidate_action=candidate_action,
                encounter_type=encounter_type,
                key_vessel_index=key_index,
                key_risk=key_vessel.risk,
                triggered=False,
                valid=True,
                response=response,
                trigger_reasons=trigger_reasons,
            )

        candidate_action = self.parse_action(response)
        valid = self._validate_action(candidate_action, encounter_type, key_vessel.risk)
        final_action = self._finalize_action(candidate_action, encounter_type, key_vessel.risk, valid)
        final_valid = self._validate_action(final_action, encounter_type, key_vessel.risk)
        kdir = self._action_to_kdir(final_action)

        self._update_memory(
            encounter_type=encounter_type,
            candidate_action=candidate_action,
            final_action=final_action,
            valid=final_valid,
            response=response,
        )
        self.last_key_risk = key_vessel.risk
        self.last_encounter_type = encounter_type

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
        )

    def _build_vessels(
        self,
        risk: Sequence[float],
        distance: Sequence[float],
        bearing: Sequence[float],
        dcpa: Sequence[float],
        tcpa: Sequence[float],
    ) -> List[VesselState]:
        risk_arr = np.atleast_1d(risk)
        distance_arr = np.atleast_1d(distance)
        bearing_arr = np.atleast_1d(bearing)
        dcpa_arr = np.atleast_1d(dcpa)
        tcpa_arr = np.atleast_1d(tcpa)

        return [
            VesselState(float(r), float(d), float(b), float(dc), float(tc))
            for r, d, b, dc, tc in zip(risk_arr, distance_arr, bearing_arr, dcpa_arr, tcpa_arr)
        ]

    def _select_key_vessel(self, vessels: List[VesselState]) -> int:
        if not vessels:
            return 0
        return int(np.argmax([v.risk for v in vessels]))

    def _trigger_reasons(self, step: int, key_risk: float, encounter_type: str) -> List[str]:
        reasons: List[str] = []
        if self.trigger_mode == "always":
            return ["always"]

        if self.trigger_mode == "fixed":
            if self.last_call_step is None or step - self.last_call_step >= self.fixed_interval:
                return ["fixed_interval"]
            return []

        if key_risk >= self.risk_threshold and self.last_key_risk < self.risk_threshold:
            reasons.append("risk_threshold_crossing")
        if (
            encounter_type != self.last_encounter_type
            and self.last_call_step is not None
            and key_risk >= self.risk_threshold
        ):
            reasons.append("encounter_type_change")
        if not self.memory.last_valid:
            reasons.append("previous_action_invalid")
        return reasons

    def _classify_encounter(self, bearing: float) -> str:
        """Classify encounter type using the existing relative-bearing convention."""
        bearing_deg = float(np.degrees(np.arctan2(np.sin(bearing), np.cos(bearing))))
        abs_bearing = abs(bearing_deg)
        if abs_bearing <= 6:
            return "head_on"
        if 6 < bearing_deg <= 112:
            return "crossing_give_way"
        if -118 <= bearing_deg < -6:
            return "crossing_stand_on"
        return "overtaking"

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
    ) -> bool:
        if not self.enable_validator:
            return action is not None
        if action is None:
            return False
        if action == Action.CONTINUE:
            return self.enable_memory and self.memory.last_executed_action is not None
        if key_risk < self.risk_threshold:
            return action in {Action.STAND_ON, Action.CONTINUE, Action.STARBOARD, Action.PORT}
        if encounter_type == "head_on":
            return action in {Action.STARBOARD, Action.CONTINUE}
        if encounter_type == "crossing_give_way":
            return action in {Action.STARBOARD, Action.CONTINUE}
        if encounter_type == "crossing_stand_on":
            return action in {Action.STAND_ON, Action.CONTINUE, Action.STARBOARD}
        if encounter_type == "overtaking":
            return action in {Action.STARBOARD, Action.PORT, Action.CONTINUE}
        return True

    def _finalize_action(
        self,
        candidate_action: Optional[Action],
        encounter_type: str,
        key_risk: float,
        valid: bool,
    ) -> Action:
        if valid and candidate_action == Action.CONTINUE:
            if self.enable_memory and self.memory.last_executed_action is not None:
                return self.memory.last_executed_action
            return self._default_safe_action(encounter_type)
        if valid and candidate_action is not None:
            return candidate_action
        if (
            self.enable_memory
            and self.memory.last_executed_action is not None
            and self._validate_action(self.memory.last_executed_action, encounter_type, key_risk)
        ):
            return self.memory.last_executed_action
        return self._default_safe_action(encounter_type)

    def _default_safe_action(self, encounter_type: str) -> Action:
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
