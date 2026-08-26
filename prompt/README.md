# Prompt Files

`prompt.txt` is the active prompt file used by the program.

## Ablation Prompts

- `prompt_none.txt`: no domain-specific COLREGs guidance, only output format
- `prompt_weak.txt`: weak maritime guidance without explicit COLREG rule framing
- `prompt_partial.txt`: partial COLREGs guidance with encounter-type awareness
- `prompt_full.txt`: full COLREGs-oriented prompt with explicit rule-aware decision logic

## Style Prompts

- `prompt_conservative.txt`: safety-first, early-action style
- `prompt_balanced.txt`: balanced safety-efficiency style
- `prompt_efficient.txt`: efficiency-first, minimum-intervention style
- `prompt_stability.txt`: maneuver-continuity and anti-oscillation style
- `prompt_aggressive.txt`: delayed-intervention, efficiency-biased style
- `prompt_rule_strict.txt`: strict rule-interpretation style
- `prompt_early_action.txt`: early decisive maneuver style

## Suggested Ablation Order

1. Copy `prompt_none.txt` to `prompt.txt`
2. Run experiments
3. Copy `prompt_weak.txt` to `prompt.txt`
4. Run experiments
5. Copy `prompt_partial.txt` to `prompt.txt`
6. Run experiments
7. Copy `prompt_full.txt` to `prompt.txt`
8. Run experiments
