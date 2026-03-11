from mesa import Model
from mesa.datacollection import DataCollector
import numpy as np

from model.resource_pool import ResourcePool
from agents.urban_agent import UrbanAgent
from modules.interaction_network import create_interaction_network
from modules.stability_analyzer import StabilityAnalyzer
from modules.shock_module import ShockModule
from model.policy_engine import PolicyEngine
from config import CONFIG


class UrbanStabilityModel(Model):

    def __init__(self, config=None):
        super().__init__()

        # Load configuration
        self.config = config if config is not None else CONFIG

        np.random.seed(self.config["random_seed"])

        self.num_agents = self.config["num_agents"]
        self.total_supply = self.config["total_supply"]

        print("Total supply at model start:", self.total_supply)

        # Resource system
        self.resource_pool = ResourcePool(self.total_supply)

        # Simulation control
        self.current_step = 0
        self.running = True

        # Create agents
        self.agents_list = []
        for _ in range(self.num_agents):
            agent = UrbanAgent(self)
            self.agents_list.append(agent)

        # Policy Engine
        self.policy_engine = PolicyEngine(self.config, self.agents_list)

        # Interaction network
        self.network = create_interaction_network(
            self.num_agents,
            self.config["avg_degree"],
            seed=self.config["random_seed"]
        )

        # Stability Analyzer
        self.stability_analyzer = StabilityAnalyzer(self.config)

        # Shock Module
        self.shock_module = ShockModule(self.config)

        # Metrics
        self.current_usi = 0.0
        self.current_gini = 0.0
        self.current_S_R = 0.0
        self.current_C = 0.0
        self.current_S_I = 0.0
        self.current_S_O = 0.0

        # Collapse detection
        self._collapse_streak = 0

        # Data Collector
        self.datacollector = DataCollector(
            model_reporters={
                "USI": lambda m: m.current_usi,
                "Gini": lambda m: m.current_gini,
                "Average_Trust": lambda m: m.current_C,
                "S_R": lambda m: m.current_S_R,
                "S_I": lambda m: m.current_S_I,
                "S_O": lambda m: m.current_S_O,
                "Total_Demand": lambda m: sum(a.demand for a in m.agents_list),
                "Total_Supply": lambda m: m.resource_pool.total_supply,
            }
        )

    # ---------------------------------------------------------
    # MAIN SIMULATION STEP
    # ---------------------------------------------------------
    def step(self):

        if not self.running:
            return

        # 1. APPLY SHOCK (if configured for this timestep)
        self.shock_module.apply(self)

        # 2. POLICY: subsidy eligibility
        self.policy_engine.compute_subsidy_eligibility()

        # 3. AGENTS compute demand
        for agent in self.agents_list:
            agent.compute_demand()

        # 4. RESOURCE ALLOCATION
        self.resource_pool.allocate(self.agents_list)

        # 5. POLICY: consumption cap
        self.policy_engine.apply_consumption_cap(self.agents_list)

        # 6. AGENTS update trust
        for agent in self.agents_list:
            agent.update_trust()

        # 7. COMPUTE STABILITY METRICS
        (
            self.current_usi,
            self.current_gini,
            self.current_S_R,
            self.current_C,
            self.current_S_I,
            self.current_S_O
        ) = self.stability_analyzer.compute_usi(
            self.agents_list,
            self.resource_pool.total_supply
        )

        # 8. COLLAPSE DETECTION
        threshold = self.config["collapse_threshold"]

        if self.current_usi < threshold:
            self._collapse_streak += 1
        else:
            self._collapse_streak = 0

        consecutive = self.config.get("collapse_consecutive_steps", 3)

        if self._collapse_streak >= consecutive:
            print(
                f"System collapsed at step {self.current_step} "
                f"(USI < {threshold} for {consecutive} steps)"
            )
            self.running = False

        # 9. COLLECT DATA
        self.datacollector.collect(self)

        # 10. ADVANCE TIME
        self.current_step += 1