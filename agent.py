class Agent:
    """
    Minimal agent interface for evolution target.

    This baseline uses random actions from the provided action_space.
    Evolving commits can replace this implementation with a learned policy.
    """

    def __init__(self, action_space, observation_space=None, config=None):
        self.action_space = action_space
        self.observation_space = observation_space
        self.config = config or {}

    def reset(self):
        # No internal state for the random policy
        return None

    def act(self, observation):
        return self.action_space.sample()

    def close(self):
        return None


