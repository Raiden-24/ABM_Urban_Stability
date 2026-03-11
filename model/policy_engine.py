import numpy as np


class PolicyEngine:
    """
    Manages and applies all configurable policy mechanisms.
    Policies are toggled via config["policy"] dict flags.
    """

    def __init__(self, config, agents_list):
        self.config = config
        self.agents_list = agents_list
        self._subsidy_set = set()  # agent indices eligible for subsidy

    def compute_subsidy_eligibility(self):
        """
        Identify bottom X% income agents for targeted subsidy.
        Called once per step before demand computation.
        """
        policy = self.config["policy"]["subsidy"]
        if not policy["enabled"]:
            self._subsidy_set = set()
            return

        percentile = policy["threshold_percentile"]
        incomes = [a.income for a in self.agents_list]
        threshold = np.percentile(incomes, percentile)

        self._subsidy_set = {
            i for i, a in enumerate(self.agents_list)
            if a.income <= threshold
        }

    def apply_pricing(self, base_price, agent):
        """
        Returns effective price after applying:
          1. Pricing multiplier (if enabled)
          2. Targeted subsidy (if enabled and agent qualifies)
        """
        effective_price = base_price

        # 1. Pricing multiplier
        pm = self.config["policy"]["pricing_multiplier"]
        if pm["enabled"]:
            effective_price *= pm["value"]

        # 2. Targeted subsidy for low-income agents
        sub = self.config["policy"]["subsidy"]
        if sub["enabled"]:
            agent_index = self.agents_list.index(agent)
            if agent_index in self._subsidy_set:
                effective_price *= (1 - sub["rate"])

        return max(effective_price, 1e-6)  # guard against zero price

    def apply_consumption_cap(self, agents):
        """
        Cap each agent's allocation after resource pool distributes.
        Excess is discarded (not redistributed).
        """
        cap_policy = self.config["policy"]["consumption_cap"]
        if not cap_policy["enabled"]:
            return

        cap_value = cap_policy["cap_value"]
        for agent in agents:
            agent.allocated = min(agent.allocated, cap_value)
            agent.allocated = max(0.0, agent.allocated)  # safety guard