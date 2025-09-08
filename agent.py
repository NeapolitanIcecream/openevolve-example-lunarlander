import math

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

        # Load defaults and apply user overrides
        defaults = self.default_config()
        user_cfg = config or {}
        self.cfg = {**defaults, **user_cfg}

        # Internal state for smoothing & episode-local memory
        self._last_action = 0
        self._switch_cooldown = 0  # steps remaining to allow next switch
        self._bank_active = False

    @classmethod
    def default_config(cls):
        """Return a dict of default hyperparameters.

        Exposed for external consumers (e.g., Optuna objective) to start from a
        known-good baseline and selectively override fields.
        """
        return {
            # Map horizontal state -> desired body angle
            "x_to_angle_k_p": 0.50,   # rad per meter
            "vx_to_angle_k_d": 1.00,  # rad per (m/s)
            "angle_limit_rad": 0.40,  # clamp desired angle

            # Altitude-scheduled horizontal mapping and angle limits
            "y_sched_low": 0.30,
            "y_sched_high": 1.40,
            "x_to_angle_k_p_low": 0.25,
            "x_to_angle_k_p_high": 0.65,
            "vx_to_angle_k_d_low": 0.60,
            "vx_to_angle_k_d_high": 1.10,
            "angle_limit_low": 0.20,
            "angle_limit_high": 0.45,

            # PD on angle tracking
            "angle_err_k_p": 0.50,
            "ang_vel_k_d": 1.00,

            # Hover/vertical regulation
            "hover_k_p": 0.50,        # position -> thrust demand proxy
            "vy_k_d": 0.50,           # velocity damping
            "hover_x_gain": 0.55,     # target height proportional to |x|

            # Altitude-scheduled vertical gains
            "hover_k_p_low": 0.65,
            "hover_k_p_high": 0.45,
            "vy_k_d_low": 0.80,
            "vy_k_d_high": 0.40,

            # Glide-slope augmentation
            "glide_vx_gain": 0.35,    # add |vx| contribution to height target
            "descent_vy_max_low": -0.20,  # near ground: gentle sink
            "descent_vy_max_high": -0.90, # high altitude: faster sink allowed
            "near_ground_y": 0.25,
            "envelope_margin": 0.05,

            # Deadbands to reduce unnecessary firings
            "angle_deadband": 0.05,   # rad
            "hover_deadband": 0.05,   # arbitrary thrust proxy units
            # Altitude-scheduled deadbands (bigger near ground to reduce jitter)
            "angle_deadband_low": 0.08,
            "angle_deadband_high": 0.05,
            "hover_deadband_low": 0.06,
            "hover_deadband_high": 0.05,

            # Sticky action (hold steps after a switch)
            "sticky_action_hold": 1,  # 0 disables

            # Contact lateral brake on ground
            "contact_lateral_brake": True,
            "contact_vx_enter": 0.18,
            "contact_vx_exit": 0.10,
            "predict_horizon": 0.30,
            "pad_half_width": 0.22,
            "pad_margin": 0.03,
            "center_tolerance": 0.05,
            "contact_brake_hold": 2,
            "contact_allow_main_for_brake": False,
            "contact_max_angle_rad": 0.12,

            # Banked side-slip approach (enable keeping beneficial bank while using main engine)
            "bank_mode_enabled": True,
            "bank_ratio_on": 0.60,      # |x|/max(y, y_min) to turn bank mode on
            "bank_ratio_off": 0.40,     # hysteresis off threshold
            "y_min_for_ratio": 0.15,    # floor for height in ratio
            "bank_deadband_scale": 2.0, # enlarge angle deadband under bank gating
            "bank_hold_steps": 3,       # sticky hold for main engine when banking
            "x_far_for_main_bias": 0.35,
            # Angle limits for bank mode (altitude scheduled)
            "angle_limit_bank_low": 0.30,
            "angle_limit_bank_high": 0.55,
            # Vector-accel mapping gains
            "kx_vec": 0.90,
            "kvx_vec": 1.20,
            "ky_vec": 0.50,
            "kvy_vec": 0.50,
            
            # Final approach near-ground hover alignment
            "final_approach_enabled": True,
            "final_y": 0.30,
            "align_x_tol": 0.05,
            "align_vx_tol": 0.15,
            "align_angle_tol": 0.05,
            "align_angvel_tol": 0.25,
            "align_gamma": 1.5,
            "align_vy_target_min": -0.05,
            "align_vy_target_max": -0.22,
            "align_bias_hover": 0.06,
            "vy_deadband": 0.02,
            "final_predict_horizon": 0.40,
            "final_main_hold_steps": 3,
            "final_offpad_score_scale": 0.70,
            "final_ascent_guard": 0.02,
        }

    def get_config(self):
        """Return a shallow copy of the current configuration."""
        return dict(self.cfg)

    def update_config(self, updates):
        """Update configuration fields in-place with given dict."""
        if not isinstance(updates, dict):
            return
        self.cfg.update(updates)

    @classmethod
    def default_search_space(cls):
        """A conservative Optuna-compatible search space spec.

        Returned format (per key):
        {"type": "float|int|bool|categorical", "low": x, "high": y, "step": s?, "choices": [...]?}
        """
        return {
            # Horizontal mapping & PD
            "x_to_angle_k_p_low": {"type": "float", "low": 0.10, "high": 0.60, "step": 0.01},
            "x_to_angle_k_p_high": {"type": "float", "low": 0.40, "high": 1.00, "step": 0.01},
            "vx_to_angle_k_d_low": {"type": "float", "low": 0.30, "high": 1.20, "step": 0.01},
            "vx_to_angle_k_d_high": {"type": "float", "low": 0.80, "high": 1.60, "step": 0.01},
            "angle_limit_low": {"type": "float", "low": 0.10, "high": 0.35, "step": 0.01},
            "angle_limit_high": {"type": "float", "low": 0.35, "high": 0.70, "step": 0.01},
            "angle_err_k_p": {"type": "float", "low": 0.20, "high": 1.20, "step": 0.01},
            "ang_vel_k_d": {"type": "float", "low": 0.40, "high": 1.80, "step": 0.01},

            # Vertical/hover
            "hover_k_p_low": {"type": "float", "low": 0.40, "high": 1.20, "step": 0.01},
            "hover_k_p_high": {"type": "float", "low": 0.20, "high": 0.90, "step": 0.01},
            "vy_k_d_low": {"type": "float", "low": 0.40, "high": 1.20, "step": 0.01},
            "vy_k_d_high": {"type": "float", "low": 0.20, "high": 0.90, "step": 0.01},
            "hover_x_gain": {"type": "float", "low": 0.20, "high": 1.20, "step": 0.01},
            "glide_vx_gain": {"type": "float", "low": 0.00, "high": 0.80, "step": 0.01},

            # Envelope & deadbands
            "descent_vy_max_low": {"type": "float", "low": -0.40, "high": -0.05, "step": 0.01},
            "descent_vy_max_high": {"type": "float", "low": -1.20, "high": -0.40, "step": 0.01},
            "envelope_margin": {"type": "float", "low": 0.02, "high": 0.10, "step": 0.005},
            "angle_deadband_low": {"type": "float", "low": 0.03, "high": 0.12, "step": 0.005},
            "angle_deadband_high": {"type": "float", "low": 0.03, "high": 0.10, "step": 0.005},
            "hover_deadband_low": {"type": "float", "low": 0.03, "high": 0.12, "step": 0.005},
            "hover_deadband_high": {"type": "float", "low": 0.03, "high": 0.10, "step": 0.005},

            # Bank mode
            "bank_mode_enabled": {"type": "categorical", "choices": [True, False]},
            "bank_ratio_on": {"type": "float", "low": 0.40, "high": 0.90, "step": 0.01},
            "bank_ratio_off": {"type": "float", "low": 0.20, "high": 0.70, "step": 0.01},
            "y_min_for_ratio": {"type": "float", "low": 0.05, "high": 0.40, "step": 0.01},
            "bank_deadband_scale": {"type": "float", "low": 1.0, "high": 3.0, "step": 0.05},
            "bank_hold_steps": {"type": "int", "low": 0, "high": 5, "step": 1},
            "x_far_for_main_bias": {"type": "float", "low": 0.20, "high": 0.60, "step": 0.01},
            "angle_limit_bank_low": {"type": "float", "low": 0.20, "high": 0.50, "step": 0.01},
            "angle_limit_bank_high": {"type": "float", "low": 0.40, "high": 0.80, "step": 0.01},
            "kx_vec": {"type": "float", "low": 0.40, "high": 1.50, "step": 0.01},
            "kvx_vec": {"type": "float", "low": 0.60, "high": 2.00, "step": 0.01},
            "ky_vec": {"type": "float", "low": 0.20, "high": 1.20, "step": 0.01},
            "kvy_vec": {"type": "float", "low": 0.20, "high": 1.20, "step": 0.01},

            # Final approach
            "final_approach_enabled": {"type": "categorical", "choices": [True, False]},
            "final_y": {"type": "float", "low": 0.15, "high": 0.60, "step": 0.01},
            "align_x_tol": {"type": "float", "low": 0.02, "high": 0.10, "step": 0.001},
            "align_vx_tol": {"type": "float", "low": 0.05, "high": 0.30, "step": 0.005},
            "align_angle_tol": {"type": "float", "low": 0.02, "high": 0.12, "step": 0.001},
            "align_angvel_tol": {"type": "float", "low": 0.10, "high": 0.60, "step": 0.01},
            "align_gamma": {"type": "float", "low": 0.8, "high": 3.0, "step": 0.05},
            "align_vy_target_min": {"type": "float", "low": -0.10, "high": -0.00, "step": 0.005},
            "align_vy_target_max": {"type": "float", "low": -0.40, "high": -0.10, "step": 0.005},
            "align_bias_hover": {"type": "float", "low": 0.00, "high": 0.20, "step": 0.005},
            "vy_deadband": {"type": "float", "low": 0.00, "high": 0.05, "step": 0.001},
            "final_predict_horizon": {"type": "float", "low": 0.10, "high": 0.80, "step": 0.01},
            "final_main_hold_steps": {"type": "int", "low": 0, "high": 6, "step": 1},
            "final_offpad_score_scale": {"type": "float", "low": 0.20, "high": 1.00, "step": 0.01},
            "final_ascent_guard": {"type": "float", "low": 0.00, "high": 0.08, "step": 0.005},

            # Contact brake
            "contact_lateral_brake": {"type": "categorical", "choices": [True, False]},
            "contact_vx_enter": {"type": "float", "low": 0.10, "high": 0.40, "step": 0.01},
            "contact_vx_exit": {"type": "float", "low": 0.05, "high": 0.30, "step": 0.01},
            "contact_brake_hold": {"type": "int", "low": 0, "high": 5, "step": 1},
        }

    @classmethod
    def suggest_config_from_trial(cls, trial, space=None):
        """Build a config dict by querying an Optuna trial.

        The function avoids importing optuna; it only relies on the trial API
        shape (suggest_float/int/categorical). You may pass a custom `space`
        spec (same structure as `default_search_space`).
        """
        spec = space if isinstance(space, dict) else cls.default_search_space()

        cfg = cls.default_config()
        for key, meta in spec.items():
            t = meta.get("type")
            if t == "float":
                low_v = meta.get("low")
                high_v = meta.get("high")
                if low_v is None or high_v is None:
                    continue
                step_v = meta.get("step")
                low_f = float(low_v)
                high_f = float(high_v)
                if step_v is None:
                    value = trial.suggest_float(key, low_f, high_f)
                else:
                    value = trial.suggest_float(key, low_f, high_f, step=float(step_v))
            elif t == "int":
                low_v = meta.get("low")
                high_v = meta.get("high")
                if low_v is None or high_v is None:
                    continue
                step_v = meta.get("step")
                step_i = int(step_v) if step_v is not None else 1
                low_i = int(low_v)
                high_i = int(high_v)
                value = trial.suggest_int(key, low_i, high_i, step=step_i)
            elif t == "categorical":
                choices = list(meta.get("choices", []))
                if not choices:
                    continue
                value = trial.suggest_categorical(key, choices)
            else:
                # Unknown type; skip
                continue
            cfg[key] = value
        return cfg

    @classmethod
    def from_trial(cls, action_space, observation_space=None, trial=None, overrides=None, space=None):
        """Construct an Agent with config suggested from an Optuna trial.

        - `overrides` can force certain values after the trial suggestion.
        - `space` can override the default search space.
        """
        if trial is None:
            raise ValueError("trial must be provided for from_trial()")
        cfg = cls.suggest_config_from_trial(trial, space=space)
        if isinstance(overrides, dict) and overrides:
            cfg.update(overrides)
        return cls(action_space, observation_space=observation_space, config=cfg)

    def reset(self):
        # Clear per-episode internal state
        self._last_action = 0
        self._switch_cooldown = 0
        self._bank_active = False
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

    def _apply_sticky_policy_with_hold(self, proposed_action, hold_override=None):
        """Sticky policy that allows an optional custom hold override for this call."""
        if hold_override is None:
            return self._apply_sticky_policy(proposed_action)

        try:
            hold = int(hold_override)
        except Exception:
            hold = 0

        if hold <= 0:
            self._last_action = proposed_action
            self._switch_cooldown = 0
            return proposed_action

        if proposed_action != self._last_action and self._switch_cooldown > 0:
            self._switch_cooldown -= 1
            return self._last_action

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

        # Altitude-scheduled parameters (0 near ground -> 1 high)
        y_low = float(self.cfg.get("y_sched_low", 0.3))
        y_high = float(self.cfg.get("y_sched_high", 1.4))
        if y_high > y_low:
            alt_t = (y - y_low) / (y_high - y_low)
        else:
            alt_t = 1.0
        if alt_t < 0.0:
            alt_t = 0.0
        elif alt_t > 1.0:
            alt_t = 1.0

        # Schedule horizontal mapping and angle limits
        x_to_angle_k_p = (
            self.cfg["x_to_angle_k_p_low"]
            + alt_t * (self.cfg["x_to_angle_k_p_high"] - self.cfg["x_to_angle_k_p_low"])
        )
        vx_to_angle_k_d = (
            self.cfg["vx_to_angle_k_d_low"]
            + alt_t * (self.cfg["vx_to_angle_k_d_high"] - self.cfg["vx_to_angle_k_d_low"])
        )
        angle_limit_rad = (
            self.cfg["angle_limit_low"]
            + alt_t * (self.cfg["angle_limit_high"] - self.cfg["angle_limit_low"])
        )

        # Compute desired angle from horizontal state with near-ground upright bias
        angle_target = x * x_to_angle_k_p + vx * vx_to_angle_k_d
        upright_scale = 0.5 + 0.5 * alt_t  # 0.5 near ground -> 1.0 high
        angle_target *= upright_scale
        if angle_target > angle_limit_rad:
            angle_target = angle_limit_rad
        elif angle_target < -angle_limit_rad:
            angle_target = -angle_limit_rad

        # PD control on angle tracking
        angle_err_k_p = self.cfg["angle_err_k_p"]
        ang_vel_k_d = self.cfg["ang_vel_k_d"]
        angle_error = angle_target - angle
        angle_todo = angle_error * angle_err_k_p - ang_vel * ang_vel_k_d

        # Hover/vertical control: schedule gains and augment target with glide-slope
        hover_k_p = (
            self.cfg["hover_k_p_low"]
            + alt_t * (self.cfg["hover_k_p_high"] - self.cfg["hover_k_p_low"])
        )
        vy_k_d = (
            self.cfg["vy_k_d_low"]
            + alt_t * (self.cfg["vy_k_d_high"] - self.cfg["vy_k_d_low"])
        )
        hover_x_gain = self.cfg["hover_x_gain"]
        glide_vx_gain = self.cfg["glide_vx_gain"]
        hover_target = hover_x_gain * abs(x) + glide_vx_gain * abs(vx) * alt_t
        hover_todo = (hover_target - y) * hover_k_p - vy * vy_k_d

        # If we have ground contact, keep upright and just damp vertical speed
        if (left_c > 0.0) or (right_c > 0.0):
            angle_todo = 0.0
            hover_todo = -vy * vy_k_d

            # Contact lateral brake: use side thrusters to suppress outward drift
            if bool(self.cfg.get("contact_lateral_brake", True)):
                try:
                    vx_enter = float(self.cfg.get("contact_vx_enter", 0.18))
                    vx_exit = float(self.cfg.get("contact_vx_exit", 0.10))
                    predict_h = float(self.cfg.get("predict_horizon", 0.30))
                    pad_half_w = float(self.cfg.get("pad_half_width", 0.22))
                    pad_margin = float(self.cfg.get("pad_margin", 0.03))
                    center_tol = float(self.cfg.get("center_tolerance", 0.05))
                    hold_steps = int(self.cfg.get("contact_brake_hold", 2))
                except Exception:
                    vx_enter, vx_exit, predict_h = 0.18, 0.10, 0.30
                    pad_half_w, pad_margin, center_tol = 0.22, 0.03, 0.05
                    hold_steps = 2

                x_pred = x + vx * predict_h
                risk_speed = abs(vx) > vx_enter
                risk_pad = (abs(x_pred) > max(pad_half_w - pad_margin, 0.0)) and (x * vx > 0.0)

                # Hysteresis-based exit conditions
                exit_hysteresis = (abs(vx) <= vx_exit) or (x * vx <= 0.0) or (abs(x) <= center_tol)

                # Descent envelope check (same formulation as below)
                descent_vy_max = (
                    self.cfg["descent_vy_max_low"]
                    + alt_t * (self.cfg["descent_vy_max_high"] - self.cfg["descent_vy_max_low"])
                )
                envelope_margin = self.cfg["envelope_margin"]
                too_fast_down_contact = vy < (descent_vy_max - envelope_margin)

                if (risk_speed or risk_pad) and (not exit_hysteresis) and (not too_fast_down_contact):
                    # Map drift direction to corrected side thruster (vx>0 -> action 1; vx<0 -> action 3)
                    action = 1 if vx > 0.0 else 3
                    action = self._safe_action(action)
                    return self._apply_sticky_policy_with_hold(action, hold_steps)

        # Banked vector-angle fusion (airborne only)
        if bool(self.cfg.get("bank_mode_enabled", True)):
            try:
                r_on = float(self.cfg.get("bank_ratio_on", 0.60))
                r_off = float(self.cfg.get("bank_ratio_off", 0.40))
                y_min_ratio = float(self.cfg.get("y_min_for_ratio", 0.15))
                kx_vec = float(self.cfg.get("kx_vec", 0.90))
                kvx_vec = float(self.cfg.get("kvx_vec", 1.20))
                ky_vec = float(self.cfg.get("ky_vec", 0.50))
                kvy_vec = float(self.cfg.get("kvy_vec", 0.50))
                angle_limit_bank = (
                    self.cfg["angle_limit_bank_low"]
                    + alt_t * (self.cfg["angle_limit_bank_high"] - self.cfg["angle_limit_bank_low"])
                )
            except Exception:
                r_on, r_off, y_min_ratio = 0.60, 0.40, 0.15
                kx_vec, kvx_vec, ky_vec, kvy_vec = 0.90, 1.20, 0.50, 0.50
                angle_limit_bank = 0.45

            # Hysteresis activation on |x|/height ratio
            denom = r_on - r_off
            if denom <= 1e-6:
                denom = 1e-6
            r = abs(x) / max(y, y_min_ratio)
            if (not self._bank_active) and (r >= r_on):
                self._bank_active = True
            elif self._bank_active and (r <= r_off):
                self._bank_active = False

            ramp = (r - r_off) / denom
            if ramp < 0.0:
                ramp = 0.0
            elif ramp > 1.0:
                ramp = 1.0
            w_bank = ramp * (0.5 + 0.5 * alt_t) if self._bank_active else 0.0

            # Vector-acceleration desired angle
            a_des_x = kx_vec * x + kvx_vec * vx
            a_des_y = ky_vec * (hover_target - y) - kvy_vec * vy
            if a_des_y < 1e-6:
                a_des_y = 1e-6
            angle_vec = math.atan2(a_des_x, a_des_y)

            # Fuse baseline PD target with vector target
            fused_angle = (1.0 - w_bank) * angle_target + w_bank * angle_vec

            # Clamp to bank-scheduled angle limit
            if fused_angle > angle_limit_bank:
                fused_angle = angle_limit_bank
            elif fused_angle < -angle_limit_bank:
                fused_angle = -angle_limit_bank

            # Recompute PD control with fused target
            angle_error = fused_angle - angle
            angle_todo = angle_error * angle_err_k_p - ang_vel * ang_vel_k_d
            angle_target = fused_angle

        # Decision logic with altitude-scheduled deadbands and descent envelope
        angle_deadband = (
            self.cfg["angle_deadband_low"]
            + alt_t * (self.cfg["angle_deadband_high"] - self.cfg["angle_deadband_low"])
        )
        hover_deadband = (
            self.cfg["hover_deadband_low"]
            + alt_t * (self.cfg["hover_deadband_high"] - self.cfg["hover_deadband_low"])
        )

        descent_vy_max = (
            self.cfg["descent_vy_max_low"]
            + alt_t * (self.cfg["descent_vy_max_high"] - self.cfg["descent_vy_max_low"])
        )
        envelope_margin = self.cfg["envelope_margin"]
        too_fast_down = vy < (descent_vy_max - envelope_margin)

        near_ground = y < float(self.cfg["near_ground_y"])
        vert_priority = 1.0 + (1.0 - alt_t)  # 2.0 near ground -> 1.0 high

        # Bank gating and effective deadband
        bank_deadband_scale = float(self.cfg.get("bank_deadband_scale", 2.0))
        bank_hold_steps = int(self.cfg.get("bank_hold_steps", 3))
        x_far_for_main_bias = float(self.cfg.get("x_far_for_main_bias", 0.35))
        bank_mode_enabled = bool(self.cfg.get("bank_mode_enabled", True))
        bank_gate_active = bank_mode_enabled and self._bank_active and (abs(x) > x_far_for_main_bias) and (not near_ground)

        angle_deadband_eff = angle_deadband * (bank_deadband_scale if bank_gate_active else 1.0)

        # Final approach: near-ground hover alignment gate
        final_hover_bias = 0.0
        try:
            final_enabled = bool(self.cfg.get("final_approach_enabled", True))
            final_y = float(self.cfg.get("final_y", self.cfg.get("near_ground_y", 0.25)))
        except Exception:
            final_enabled = True
            final_y = 0.30

        if final_enabled and (y < final_y) and (left_c <= 0.0 and right_c <= 0.0):
            tiny = 1e-6
            try:
                ax_tol = float(self.cfg.get("align_x_tol", 0.05))
                avx_tol = float(self.cfg.get("align_vx_tol", 0.15))
                aang_tol = float(self.cfg.get("align_angle_tol", 0.05))
                aangv_tol = float(self.cfg.get("align_angvel_tol", 0.25))
                align_gamma = float(self.cfg.get("align_gamma", 1.5))
            except Exception:
                ax_tol, avx_tol, aang_tol, aangv_tol, align_gamma = 0.05, 0.15, 0.05, 0.25, 1.5

            ex = min(abs(x) / max(ax_tol, tiny), 1.0)
            evx = min(abs(vx) / max(avx_tol, tiny), 1.0)
            ea = min(abs(angle) / max(aang_tol, tiny), 1.0)
            eav = min(abs(ang_vel) / max(aangv_tol, tiny), 1.0)
            e = math.sqrt((ex * ex + evx * evx + ea * ea + eav * eav) / 4.0)
            score = 1.0 - e
            if score < 0.0:
                score = 0.0
            elif score > 1.0:
                score = 1.0
            score_nl = score ** align_gamma

            # Optional pad alignment gate
            try:
                pred_h = float(self.cfg.get("final_predict_horizon", 0.40))
                pad_half_w = float(self.cfg.get("pad_half_width", 0.22))
                pad_margin = float(self.cfg.get("pad_margin", 0.03))
                offpad_scale = float(self.cfg.get("final_offpad_score_scale", 0.70))
            except Exception:
                pred_h, pad_half_w, pad_margin, offpad_scale = 0.40, 0.22, 0.03, 0.70
            x_pred_final = x + vx * pred_h
            if abs(x_pred_final) > max(pad_half_w - pad_margin, 0.0):
                if offpad_scale < 0.0:
                    offpad_scale = 0.0
                elif offpad_scale > 1.0:
                    offpad_scale = 1.0
                score_nl *= offpad_scale

            # Dynamic descent envelope based on alignment score
            try:
                vy_min = float(self.cfg.get("align_vy_target_min", -0.05))
                vy_max = float(self.cfg.get("align_vy_target_max", -0.22))
                vy_deadband = float(self.cfg.get("vy_deadband", 0.02))
            except Exception:
                vy_min, vy_max, vy_deadband = -0.05, -0.22, 0.02
            vy_limit_eff = vy_min + score_nl * (vy_max - vy_min)
            margin = envelope_margin if envelope_margin > vy_deadband else vy_deadband
            too_fast_down_final = vy < (vy_limit_eff - margin)

            # Hover bias to encourage holding main engine when misaligned
            try:
                align_bias_hover = float(self.cfg.get("align_bias_hover", 0.10))
            except Exception:
                align_bias_hover = 0.10
            final_hover_bias = align_bias_hover * (1.0 - score) * (1.0 - alt_t)

            # Ascent guard: if ascending near ground, avoid adding positive hover bias
            try:
                ascent_guard = float(self.cfg.get("final_ascent_guard", 0.02))
            except Exception:
                ascent_guard = 0.02
            if vy > ascent_guard:
                final_hover_bias = 0.0

            if too_fast_down_final:
                action = self._safe_action(2)
                try:
                    hold_steps = int(self.cfg.get("final_main_hold_steps", 3))
                except Exception:
                    hold_steps = 3
                return self._apply_sticky_policy_with_hold(action, hold_steps)

        action = 0  # default: do nothing
        if too_fast_down:
            action = 2  # enforce descent envelope with main engine
        else:
            vertical_drive = (hover_todo + final_hover_bias) - hover_deadband
            angle_drive_abs = abs(angle_todo) - angle_deadband_eff

            # Prefer main engine hold when banking and angle error is within enlarged deadband
            if bank_gate_active and (vertical_drive > -0.5 * hover_deadband) and (abs(angle_todo) <= angle_deadband_eff):
                action = 2
                action = self._safe_action(action)
                return self._apply_sticky_policy_with_hold(action, bank_hold_steps)

            if vertical_drive * vert_priority > max(angle_drive_abs, 0.0) and vertical_drive > 0.0:
                action = 2  # main engine
            elif angle_todo < -angle_deadband_eff:
                action = 3  # right orientation engine
            elif angle_todo > angle_deadband_eff:
                action = 1  # left orientation engine
            else:
                action = 0

        action = self._safe_action(action)
        action = self._apply_sticky_policy(action)
        return action

    def close(self):
        # No special cleanup required
        return None
