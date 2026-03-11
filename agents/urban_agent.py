from mesa import Agent
import numpy as np


class UrbanAgent(Agent):
    """
    Urban household agent.
    Attributes:
        income       : Economic capacity (lognormal distributed)
        trust        : Cooperation level (0–1)
        demand       : Requested resource amount
        allocated    : Resource actually received
        cooperating  : Boolean flag for cooperation state
    """


    def __init__(self, model):
        super().__init__(model)

        # Lognormal income distribution
        self.income = np.random.lognormal(
            mean=model.config["income_mean"],
            sigma=model.config["income_sigma"]
        )

        # Uniform trust initialization
        self.trust = np.random.uniform(
            model.config["initial_trust_range"][0],
            model.config["initial_trust_range"][1]
        )

        self.demand = 0
        self.allocated = 0
        self.cooperating = True

    def compute_demand(self):
        """
        Demand is proportional to income and trust.
        Effective price handled later in policy phase.
        """
        effective_price = self.model.policy_engine.apply_pricing(
            self.model.config["base_price"], self
        )
        self.demand = (self.income * self.trust) / effective_price

    def update_trust(self):
        """
        Trust update using network-based neighbour influence.
        Formula (from execution plan):
            trust_new = α × personal_experience + (1 - α) × mean_neighbour_trust
        where α = config["trust_weight_personal"] (default 0.7)
        """
        if self.demand == 0:
            return

        # Personal experience: satisfaction ratio (how much of demand was met)
        personal_experience = min(self.allocated / self.demand, 1.0)

        # Neighbour trust: average trust of connected agents in interaction network
        network = self.model.network
        agent_index = self.model.agents_list.index(self)

        if agent_index in network:
            neighbour_indices = list(network.neighbors(agent_index))
            if neighbour_indices:
                mean_neighbour_trust = sum(
                    self.model.agents_list[n].trust
                    for n in neighbour_indices if n < len(self.model.agents_list)
                ) / len(neighbour_indices)
            else:
                mean_neighbour_trust = self.trust  # no neighbours → use own trust
        else:
            mean_neighbour_trust = self.trust

        # Blend personal experience with neighbour influence
        alpha = self.model.config["trust_weight_personal"]  # default 0.7
        self.trust = alpha * personal_experience + (1 - alpha) * mean_neighbour_trust

        # Bound trust between 0 and 1
        self.trust = max(0.0, min(1.0, self.trust))

    def step(self):
        """
        Full agent cycle for one time step.
        """
        self.compute_demand()
        # Allocation handled in ResourcePool
        self.update_trust()