class Agent:
    """
    Heuristic PD controller for Gymnasium LunarLander-v3 (Discrete).

    - Horizontal regulation: map horizontal position/velocity (x, vx) to a desired
      body angle target, then use orientation engines (left/right) via PD control
      on (angle, angular_velocity).
    - Vertical regulation: hover controller tracks a height target that scales with
      horizontal distance to pad; main engine used when vertical demand dominates.
    - Contact logic: if any leg is in contact, keep body upright and damp vertical
      velocity without tilting.
    - Sticky action: optional short hold to suppress oscillatory switching.

    Observation (s): [x, y, vx, vy, angle, angular_velocity, left_contact, right_contact]
    Actions: 0=nop, 1=left orientation, 2=main engine, 3=right orientation
    """

    def __init__(self, action_space, observation_space=None, config=None):
        self.action_space = action_space
        self.observation_space = observation_space

        # Default hyperparameters tuned for stable landings with low jitter.
        defaults = {
            # Map horizontal state -> desired body angle
            "x_to_angle_k_p": 0.50,   # rad per meter
            "vx_to_angle_k_d": 1.00,  # rad per (m/s)
            "angle_limit_rad": 0.40,  # clamp desired angle

            # PD on angle tracking
            "angle_err_k_p": 0.50,
            "ang_vel_k_d": 1.00,

            # Hover/vertical regulation
            "hover_k_p": 0.50,        # position -> thrust demand proxy
            "vy_k_d": 0.50,           # velocity damping
            "hover_x_gain": 0.55,     # target height proportional to |x|

            # Deadbands to reduce unnecessary firings
            "angle_deadband": 0.05,   # rad
            "hover_deadband": 0.05,   # arbitrary thrust proxy units

            # Sticky action (hold steps after a switch)
            "sticky_action_hold": 1,  # 0 disables
        }

        user_cfg = config or {}
        self.cfg = {**defaults, **user_cfg}

        # Internal state for smoothing & episode-local memory
        self._last_action = 0
        self._switch_cooldown = 0  # steps remaining to allow next switch

    def reset(self):
        # Clear per-episode internal state
        self._last_action = 0
        self._switch_cooldown = 0
        return None

    def _safe_action(self, action_candidate):
        """Return a valid discrete action int within the action_space, fallback to 0."""
        try:
            a = int(action_candidate)
        except Exception:
            a = 0

        # Validate against Discrete(n) if available
        try:
            n = int(getattr(self.action_space, "n"))
            if not (0 <= a < n):
                a = 0
        except Exception:
            # If action space lacks attribute 'n', conservatively clamp to [0, 3]
            if a < 0 or a > 3:
                a = 0
        return a

    def _apply_sticky_policy(self, proposed_action):
        """Optionally hold the previous action for a few steps to reduce jitter."""
        hold = int(self.cfg.get("sticky_action_hold", 0))
        if hold <= 0:
            self._last_action = proposed_action
            self._switch_cooldown = 0
            return proposed_action

        # If different action is proposed but we are in cooldown, keep last
        if proposed_action != self._last_action and self._switch_cooldown > 0:
            self._switch_cooldown -= 1
            return self._last_action

        # Accept the proposed action and reset cooldown
        self._last_action = proposed_action
        self._switch_cooldown = hold
        return proposed_action

    def act(self, observation):
        # Basic validation
        try:
            s = list(observation)
        except Exception:
            return self._safe_action(0)

        if len(s) < 8:
            return self._safe_action(0)

        # Unpack observation
        try:
            x, y, vx, vy, angle, ang_vel, left_c, right_c = (
                float(s[0]), float(s[1]), float(s[2]), float(s[3]),
                float(s[4]), float(s[5]), float(s[6]), float(s[7])
            )
        except Exception:
            return self._safe_action(0)

        # Compute desired angle from horizontal state
        x_to_angle_k_p = self.cfg["x_to_angle_k_p"]
        vx_to_angle_k_d = self.cfg["vx_to_angle_k_d"]
        angle_limit_rad = self.cfg["angle_limit_rad"]

        angle_target = x * x_to_angle_k_p + vx * vx_to_angle_k_d
        if angle_target > angle_limit_rad:
            angle_target = angle_limit_rad
        elif angle_target < -angle_limit_rad:
            angle_target = -angle_limit_rad

        # PD control on angle tracking
        angle_err_k_p = self.cfg["angle_err_k_p"]
        ang_vel_k_d = self.cfg["ang_vel_k_d"]
        angle_error = angle_target - angle
        angle_todo = angle_error * angle_err_k_p - ang_vel * ang_vel_k_d

        # Hover/vertical control: target height grows with |x|
        hover_k_p = self.cfg["hover_k_p"]
        vy_k_d = self.cfg["vy_k_d"]
        hover_x_gain = self.cfg["hover_x_gain"]
        hover_target = hover_x_gain * abs(x)
        hover_todo = (hover_target - y) * hover_k_p - vy * vy_k_d

        # If we have ground contact, keep upright and just damp vertical speed
        if (left_c > 0.0) or (right_c > 0.0):
            angle_todo = 0.0
            hover_todo = -vy * vy_k_d

        # Decision logic with deadbands
        angle_deadband = self.cfg["angle_deadband"]
        hover_deadband = self.cfg["hover_deadband"]

        action = 0  # default: do nothing
        if hover_todo > abs(angle_todo) and hover_todo > hover_deadband:
            action = 2  # main engine
        elif angle_todo < -angle_deadband:
            action = 3  # right orientation engine
        elif angle_todo > angle_deadband:
            action = 1  # left orientation engine
        else:
            action = 0

        action = self._safe_action(action)
        action = self._apply_sticky_policy(action)
        return action

    def close(self):
        # No special cleanup required
        return None
