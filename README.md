# openevolve-example-lunarlander (Agent repository)

This is the evolving target repository for OpenEvolve (the Agent repo). The upstream env repository’s evaluator runs this repo’s `agent.py` in Gymnasium’s LunarLander environment and uses the mean episode return as the primary score (`combined_score`) to drive evolution.

## Background: LunarLander (Discrete, v3)

LunarLander is a classic control task from Gymnasium’s Box2D suite: safely land the lunar module between two flags on a flat landing pad. The task rewards stable, low-velocity, low-angle, fuel-aware landings and penalizes crashes and excessive engine usage.

- Goal: maximize episode return.
- Termination:
  - `terminated`: either a successful landing (both legs down within acceptable speed/angle) or a crash.
  - `truncated`: time/step limit reached (from wrappers or evaluator limits).

For details, see: [Gymnasium Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/).

## Interface (Observation / Action Space)

This example uses the discrete variant `LunarLander-v3`:

- Observation (shape: 8, dtype float):
  1) lander position in x, y (relative to the pad)
  2) linear velocities in x, y
  3) lander angle, angular velocity
  4) left_leg_contact, right_leg_contact (0.0 or 1.0)

- Action (Discrete(4)):
  - 0: do nothing
  - 1: fire left orientation engine
  - 2: fire main engine
  - 3: fire right orientation engine

Gymnasium also provides a continuous variant `LunarLanderContinuous-v3` (not used here).

## Reward Shaping (Discrete v3)

At each step, rewards are shaped to encourage safe and efficient landings. The episode return is the sum of per-step rewards plus any terminal bonus/penalty. Specifically (see the docs above):

- Step-wise shaping terms:
  - Closer to the landing pad → higher reward; farther → lower reward.
  - Slower movement (lower |vx|, |vy|) → higher reward; faster → lower reward.
  - Smaller tilt angle → higher reward; larger tilt (not horizontal) → penalty.
  - +10 points per leg in contact with the ground (up to +20 with both legs).
  - Engine usage penalties per frame:
    - −0.03 for each frame a side (orientation) engine is firing.
    - −0.3 for each frame the main engine is firing.

- Terminal bonus/penalty:
  - +100 for a safe landing
  - −100 for a crash

- A commonly used “solved” threshold is an episode return ≥ 200.

Reference: [Gymnasium Lunar Lander – Rewards](https://gymnasium.farama.org/environments/box2d/lunar_lander/).

## Agent Contract (Boundary)

The evaluator (env side) owns environment creation, seeding, episode count, and step limits. The Agent interacts only through the following interface, returning a valid discrete action. The Agent may keep internal state across steps; `reset()` should clear per-episode state. The Agent must not modify environment internals.

`agent.py` at the repo root must expose:

```python
class Agent:
    def __init__(self, action_space, observation_space=None, config=None):
        ...

    def reset(self):
        # Called at the start of every episode
        ...

    def act(self, observation):
        # Return an int action in {0, 1, 2, 3}
        ...

    def close(self):
        # Optional cleanup
        ...
```

## Evaluator Assumptions (for evolution scoring)

- Env ID: `LunarLander-v3`
- Episodes per evaluation: default 5
- Max steps per episode: default 1000
- Seeding: fixed base seed with per-episode offset (for stability with mild stochasticity)
- Primary metric: `combined_score = mean(episode_total_reward)`; auxiliary metrics such as `mean_reward`, `std_reward`, etc.

## Baseline

The default `agent.py` is a random policy (samples from `action_space`). OpenEvolve evolves this repository via commit-based edits to improve the mean return.

## Development Notes

- Always return a valid discrete action (int in 0–3) from `act`.
- Initialize/clear internal state in `__init__` / `reset()` as needed.
- Any training/learning logic should live inside this repo; the evaluator only runs episodes and scores them.
