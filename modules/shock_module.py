class ShockModule:
    """
    Injects configurable external shocks into the simulation.

    Supported shock types:
        "resource_scarcity" -- reduces total_supply by magnitude %
        "trust_breakdown"   -- reduces each agent's trust by magnitude fraction

    Config (from config["shock"]):
        enabled    : bool   -- whether shocks are active
        type       : str    -- "resource_scarcity" or "trust_breakdown"
        step       : int    -- time step at which to inject shock
        magnitude  : float  -- fraction of reduction (e.g. 0.3 = 30%)
        persistent : bool   -- if True, effect persists every step after t_shock
                               if False, shock applied once and supply reset next step
    """

    def __init__(self, config):
        self.config = config
        self._shock_applied = False          # tracks one-shot injection

    def apply(self, model):
        """
        Call every step. Injects shock if conditions are met.
        Returns True if a shock was active this step, else False.
        """
        shock_cfg = self.config.get("shock", {})

        if not shock_cfg.get("enabled", False):
            return False

        t_shock    = shock_cfg.get("step", 25)
        magnitude  = shock_cfg.get("magnitude", 0.3)
        shock_type = shock_cfg.get("type", "resource_scarcity")
        persistent = shock_cfg.get("persistent", False)

        # Determine if shock should fire this step
        should_shock = False

        if persistent and model.current_step >= t_shock:
            should_shock = True                  # active every step from t_shock
        elif not self._shock_applied and model.current_step == t_shock:
            should_shock = True                  # one-shot at exactly t_shock
            self._shock_applied = True

        if not should_shock:
            return False

        # --- Apply the shock ---
        if shock_type == "resource_scarcity":
            # 3.8: Reduce total supply
            original_supply = model.resource_pool.total_supply
            model.resource_pool.total_supply = original_supply * (1 - magnitude)
            model.resource_pool.available_supply = model.resource_pool.total_supply

        elif shock_type == "trust_breakdown":
            # 3.9: Reduce each agent's trust proportionally
            for agent in model.agents_list:
                agent.trust = max(0.0, agent.trust * (1 - magnitude))

        return True
