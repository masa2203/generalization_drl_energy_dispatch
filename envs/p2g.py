from typing import Union, Optional, Tuple

from envs.constants import const


class P2G:
    """
    A class to model a power-to-gas (P2G) system.

    :param total_cap: A float or integer that represents the total capacity for the stored gas in pounds (lb).
    :param min_charge_rate: A float or integer that represents the minimum required rate of charge (if on) (MW).
    :param max_charge_rate: A float or integer that represents the maximum possible rate of charge (MW).
    :param charge_eff: A float that represents the charging efficiency, including electrolyzer, methanation,
                        compression, etc. (fraction).
    :param replacement_cost: A float or integer that represents the replacement cost of the P2G system (CAD / MW).
    :param total_hours: An integer that represents the equipment lifetime (hours).
    :param price_co2: A float or integer that represents the cost of purchasing CO2 (CAD / kg).
    :param assign_cost_at_discharge: An optional boolean that represents whether cost (CO2 and degradation) are assigned
                        at gas generation or discharge. Default is False (assigned at generation).
    :param assign_lost_profits_at_discharge: An optional boolean that represents whether lost profits are assigned
                    at gas discharge. Default is False (no lost profits assignment).
    :param resolution_h: An optional float that represents the resolution in hours. Default is 1.0.
    :param tracking: An optional boolean that represents whether to track variables in step function. Default is True.
    """
    def __init__(self,
                 total_cap: Union[int, float],
                 min_charge_rate: Union[int, float],
                 max_charge_rate: Union[int, float],
                 charge_eff: float,
                 replacement_cost: Union[int, float],  # in CAD/MW
                 total_hours: int,  # equipment lifetime in hours
                 price_co2: Union[int, float],  # in CAD/kg
                 assign_cost_at_discharge: Optional[bool] = False,
                 assign_lost_profits_at_discharge: Optional[bool] = False,
                 resolution_h: float = 1.0,
                 tracking: Optional[bool] = True,
                 ):
        """
        Constructs a new P2G object.

        :param total_cap: A float or integer that represents the total capacity for the stored gas in pounds (lb).
        :param min_charge_rate: A float or integer that represents the minimum required rate of charge (if on) (MW).
        :param max_charge_rate: A float or integer that represents the maximum possible rate of charge (MW).
        :param charge_eff: A float that represents the charging efficiency, including electrolyzer, methanation,
                            compression, etc. (fraction).
        :param replacement_cost: A float or integer that represents the replacement cost of the P2G system (CAD / MW).
        :param total_hours: An integer that represents the equipment lifetime (hours).
        :param price_co2: A float or integer that represents the cost of purchasing CO2 (CAD / kg).
        :param assign_cost_at_discharge: An optional boolean that represents whether cost (CO2 and degradation) are
                                assigned at gas generation or discharge. Default is True (assigned at generation).
        :param assign_lost_profits_at_discharge: An optional boolean that represents whether lost profits are assigned
                    at gas discharge. Default is False (no lost profits assignment).
        :param resolution_h: An optional float that represents the resolution in hours. Default is 1.0.
        :param tracking: An optional boolean that represents whether to track variables in step function. Default is
        True.
        """
        assert 0 <= charge_eff <= 1, "Charge efficiency must be between 0 and 1."
        assert min_charge_rate > 0, "min_charge_rate must be greater than zero."
        assert max_charge_rate > 0, "max_charge_rate must be greater than zero."

        # ARGUMENTS
        self.total_cap = total_cap  # lb
        self.min_charge_rate = min_charge_rate  # MW
        self.max_charge_rate = max_charge_rate  # MW
        self.charge_eff = charge_eff  # fraction
        self.resolution_h = resolution_h  # resolution in hours
        self.price_co2 = price_co2
        self.tracking = tracking
        self.assign_cost_at_discharge = assign_cost_at_discharge
        self.assign_lost_profits_at_discharge = assign_lost_profits_at_discharge

        self.soc = 0

        # PRECOMPUTE
        # Degradation cost for each timestep (max_charge_rate represents installed capacity)
        self.degradation_cost_per_timestep = (replacement_cost * max_charge_rate / total_hours) * resolution_h

        # Conversion rate
        # MWh -> MJ * MJ/kg * kg/lb = lb (for CH4)
        self.mwh_to_pound = const['MWh_to_MJ'] / const['ch4_lhv'] / const['pound_to_kg']

        # TRACKERS
        self.count = 0
        self.socs = []  # tracks SOCs
        self.power_flows = []  # tracks power flows from plant view
        self.p2g_costs = []  # tracks degradation and CO2 cost sum
        self.lost_profits = []  # tracks lost profit if applicable

        self.running_cost_sum = 0
        self.running_cost_sums = []  # tracks running_cost_sum for partial reset

        self.running_lost_profits_sum = 0
        self.running_lost_profits_sums = []  # tracks running_lost_profits_sum for partial reset

    def step(self,
             action: float,
             avail_power: float,
             fuelflow: float,
             pool_price: Optional[float] = None) -> Tuple[float, float, float]:
        """
        Conducts one step with the P2G system.

        :param action: A float that represents the range(0,1) used to charge the storage.
        :param avail_power: A float that represents the max. power available for charging (MW).
        :param fuelflow: A float that represents the flow of CH4 out of the storage (pph).
        :param pool_price: An optional float that represents the pool price ($/MWh).
        :return: A tuple of three floats that represent the power flow from plant view, the degradation cost,
                and the lost_profits cost component (if applicable).
        """
        assert avail_power >= 0, "Available power must be non-negative!"
        assert fuelflow >= 0, "Fuelflow must be non-negative!"
        assert 0 <= action <= 1, f"Action out of bounds [0, 1]. Action passed: {action}"
        assert not self.assign_lost_profits_at_discharge or pool_price is not None, \
            'Must pass pool_price if lost profits should be taken into account.'

        power_flow = 0
        p2g_cost = 0
        lost_profit = 0
        # negative -> profit lost at current time step, subtract from reward to encourage P2G usage (- and - = +)
        # postive -> profit lost at previous time steps, subtract from reward to account for previous losses

        # No P2G activity in case of fuelflow from storage to GT(s)
        if fuelflow > 0:
            # Get new SOC after releasing gas
            new_soc = self.soc - ((fuelflow * self.resolution_h) / self.total_cap)
            new_soc = max(0, new_soc)

            if self.assign_cost_at_discharge or self.assign_lost_profits_at_discharge:
                # Compute the ratio of SOC reduction
                change_ratio = (self.soc - new_soc) / self.soc if self.soc > 0 else 0

                if self.assign_cost_at_discharge:
                    # Compute cost share of current gas release and update class vars
                    p2g_cost = self.running_cost_sum * change_ratio
                    self.running_cost_sum -= p2g_cost

                if self.assign_lost_profits_at_discharge:
                    # Compute cost share of current gas release and update class vars
                    lost_profit = abs(self.running_lost_profits_sum * change_ratio)
                    self.running_lost_profits_sum += lost_profit

            # Update SOC
            self.soc = new_soc

        # Check on P2G operation
        elif action > 0:
            # Bounds of charge rate
            rate = action * self.max_charge_rate
            if rate >= self.min_charge_rate and avail_power >= self.min_charge_rate:
                energy_flow, p2g_cost = self._charge(rate=rate, avail_power=avail_power)
                power_flow = energy_flow / self.resolution_h  # power flow from plant perspective
                if self.assign_cost_at_discharge:
                    self.running_cost_sum += p2g_cost
                    p2g_cost = 0
                if self.assign_lost_profits_at_discharge:
                    lost_profit = (power_flow * pool_price)  # value of e-power put into P2G system
                    self.running_lost_profits_sum += lost_profit

        if self.tracking:
            self._tracking(power_flow, p2g_cost, lost_profit)

        self.count += 1

        return power_flow, p2g_cost, lost_profit

    def _charge(self, rate: float, avail_power: float) -> float:
        """
        Operates the P2G system.

        Notes:
        - Charges with desired rate unless less power is available
        - Charges with desired rate unless max capacity is reached
        - Efficiencies are reflected through lower SOC after charging

        :param rate: A float that represents the charge rate in MW.
        :param avail_power: A float that represents the available power for charging in MW.
        :return: A tuple: (float that represents the energy used to charge the storage,
                            float that represents the cost of degradation and co2 purchases).
        """
        # Pick min of desired charge rate and available power, multiply by time to obtain energy
        available_energy = min(rate, avail_power) * self.resolution_h

        # Max. added fuel (CH4) in pound
        added_fuel = available_energy * self.charge_eff * self.mwh_to_pound

        # Update SOC
        old_soc = self.soc
        new_soc = self.soc + added_fuel / self.total_cap
        self.soc = min(new_soc, 1)  # avoids overcharging

        # Compute effective energy flow, negative sign to get flow from plant view
        effective_mass_gain = (self.soc - old_soc) * self.total_cap
        energy_flow = -(effective_mass_gain / (self.mwh_to_pound * self.charge_eff))

        # Compute CO2 cost, then add degradation cost
        # CO2 in CAD/kg, mass gain in lbs, 2.7kg of CO2 needed per kg of produced CH4
        mass_gain_kg = effective_mass_gain * const['pound_to_kg']
        co2_cost = self.price_co2 * mass_gain_kg * 2.7

        p2g_cost = co2_cost + self.degradation_cost_per_timestep

        return energy_flow, p2g_cost

    def _tracking(self, power_flow: float, p2g_cost: float, lost_profit: float):
        """
        Keeps track of storage behavior over time.

        :param power_flow: A float that represents the energy flow.
        :param p2g_cost: A float that represents the sum of degradation and CO2 cost.
        :param lost_profit: A float that represents the lost profits at the current time step.
        """
        self.socs.append(self.soc)
        self.power_flows.append(round(power_flow, 2))
        self.p2g_costs.append(round(p2g_cost, 2))
        self.lost_profits.append(round(lost_profit, 2))

        if self.assign_cost_at_discharge:
            self.running_cost_sums.append(round(self.running_cost_sum, 2))

        if self.assign_lost_profits_at_discharge:
            self.running_lost_profits_sums.append(round(self.running_lost_profits_sum, 2))

    def reset(self, options: Optional[dict] = None):
        """
        Resets the storage.

        :param options: An optional dictionary that determines the reset options.
        """
        # Define default options and overwrite if options are passed
        default_options = {'tracking': self.tracking}

        if options is not None:
            default_options.update(options)

        # Update tracking parameter
        self.tracking = default_options['tracking']

        self.count = 0
        self.soc = 0
        self.socs = []
        self.power_flows = []
        self.p2g_costs = []
        self.lost_profits = []
        self.running_cost_sum = 0
        self.running_cost_sums = []
        self.running_lost_profits_sum = 0
        self.running_lost_profits_sums = []

    def partial_reset(self, n):
        """
        Resets the storage partially.

        :param n: An integer that represents the number of steps to reset.
        """
        if self.count > n:
            self.count -= n
            self.socs = self.socs[:-n]
            self.power_flows = self.power_flows[:-n]
            self.p2g_costs = self.p2g_costs[:-n]
            self.lost_profits = self.lost_profits[:-n]

            self.soc = self.socs[-1]

            if self.assign_cost_at_discharge:
                self.running_cost_sums = self.running_cost_sums[:-n]
                self.running_cost_sum = self.running_cost_sums[-1]

            if self.assign_lost_profits_at_discharge:
                self.running_lost_profits_sums = self.running_lost_profits_sums[:-n]
                self.running_lost_profits_sum = self.running_lost_profits_sums[-1]
        else:
            self.reset()
