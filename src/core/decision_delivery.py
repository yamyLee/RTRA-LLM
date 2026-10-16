"""Virtual-time delivery of synchronous LLM responses for latency experiments.

The API call may block wall time. Its response is queued in simulated time while
ship motion continues under the last accepted instruction. At most one request
is in flight; expired/failed responses never become active commands.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class PendingDecision:
    decision: object
    requested_step: int
    ready_step: int
    status: str
    event: dict


class DecisionDelivery:
    def __init__(self, args, dt):
        self.dt = float(dt)
        self.mode = args.llm_latency_mode
        self.base = float(args.llm_latency_s)
        self.jitter = float(args.llm_latency_jitter_s)
        self.timeout = float(args.llm_timeout_s)
        self.max_age = float(args.llm_max_decision_age_s)
        self.failure_probability = float(args.llm_failure_probability)
        self.retry_steps = max(1, self.steps(args.llm_retry_interval_s))
        values = [self.base, self.jitter, self.timeout, self.max_age, args.llm_retry_interval_s]
        if self.mode not in {'zero', 'fixed', 'random', 'measured'}:
            raise ValueError('Unknown LLM latency mode')
        if not np.all(np.isfinite(values)) or min(values) < 0 or self.timeout <= 0:
            raise ValueError('Latency values must be finite and nonnegative; timeout must be positive')
        if not 0 <= self.failure_probability <= 1:
            raise ValueError('Failure probability must be in [0, 1]')
        # Timing/failure draws never consume the vessel disturbance RNG.
        self.rng = np.random.default_rng(np.random.SeedSequence([int(args.seed or 0), 173]))
        self.pending = None
        self.active = None
        self.delivered_event = None
        self.next_request_step = 0
        self.discarded_count = 0

    def steps(self, seconds):
        return max(0, int(math.ceil(float(seconds) / self.dt - 1e-10)))

    def advance(self, step, supervisor, observation):
        self.delivered_event = None
        if self.pending is not None or step < self.next_request_step:
            supervisor.observe(step=step, **observation)
        if self.pending is not None and step >= self.pending.ready_step:
            packet = self.pending
            self.pending = None
            age = (step - packet.requested_step) * self.dt
            status = packet.status
            if status == 'ready' and age > self.max_age + 1e-10:
                status = 'discarded_stale'
            packet.event.update(delivery_step=step, decision_age_s=age)
            if status == 'ready':
                self.active = supervisor.revalidate(packet.decision)
                supervisor.memory.last_valid = self.active.valid
                supervisor.memory.last_planned_action = self.active.final_action
                if not self.active.valid:
                    self.next_request_step = step + self.retry_steps
                packet.event.update(delivery_status='accepted' if self.active.valid else 'accepted_fallback',
                    effective_step=step, revalidated_action=self.active.final_action.value)
                self.delivered_event = packet.event
            else:
                packet.event.update(delivery_status=status, effective_step=None)
                self.discarded_count += 1
                supervisor.memory.last_valid = False
                self.next_request_step = step + self.retry_steps
            # Do not issue a second query on the same delivery tick.
            return supervisor.revalidate(self.active) if self.active else None
        if self.pending is not None or step < self.next_request_step:
            return supervisor.revalidate(self.active) if self.active else None
        decision = supervisor.update(step=step, **observation)
        if not decision.triggered or supervisor.rule_only:
            self.active = decision
            if decision.triggered and supervisor.trigger_history:
                self.delivered_event = supervisor.trigger_history[-1]
                self.delivered_event.update(delivery_status='accepted_rule_decision',effective_step=step)
            return decision
        event = supervisor.trigger_history[-1]
        delay = (0.0 if self.mode == 'zero' else self.base if self.mode == 'fixed' else
                 max(0., self.base + self.rng.uniform(-self.jitter, self.jitter))
                 if self.mode == 'random' else float(event.get('inference_latency_s', 0.)))
        failed = not event.get('call_ok', True)
        injected = self.rng.random() < self.failure_probability
        status = 'discarded_call_failure' if failed else 'discarded_injected_failure' if injected else 'ready'
        due = delay
        if delay > self.timeout:
            due, status = self.timeout, 'discarded_timeout'
        event.update(request_step=step, simulated_latency_s=delay, delivery_status='pending', effective_step=None)
        self.pending = PendingDecision(decision, step, step + self.steps(due), status, event)
        if self.pending.ready_step == step:
            # Refresh against the current state before immediate delivery too.
            return self.advance(step, supervisor, observation)
        return supervisor.revalidate(self.active) if self.active else None
