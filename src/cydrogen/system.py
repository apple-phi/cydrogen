import numpy as np
from .graph import Graph
from .units import HOUR, WH, HOURS_PER_YEAR
from .model import (
    BatteryConnectionToElectrolyserCompressorProcess,
    EnergyStore,
    Battery,
    H2Refueler,
    PVProcess,
    PVProcessToBatteryConnection,
    Sun,
    add_energy_sinks,
)


# NOTE: currently assumes user has constructed the graph using dummy processes as necessary
# the need for dummy processes is analogous to critical path analysis.
class EnergySystem:
    def __init__(self, g: Graph, dt=HOUR) -> None:
        self.g = g
        self.dt = dt
        self.U_max = np.array(g.node_apply(lambda n: n.max_energy_stored))
        self.U_frac_dt_loss = dt * np.array(
            g.node_apply(lambda n: n.static_frac_power_loss)
        )
        self.A = g.get_adjacency_matrix()
        self.weighted_A = g.get_adjacency_matrix(lambda e: e.weight * e.efficiency)
        self.process_power_limits = g.get_adjacency_matrix(
            lambda e: e.max_power_transfer
        )
        # print(f"U_max:\n{self.U_max}")
        # print(f"U_frac_dt_loss:\n{self.U_frac_dt_loss}")
        # print(f"A:\n{self.A}")
        # print(f"weighted_A:\n{self.weighted_A}")
        # print(f"process_power_limits:\n{self.process_power_limits}")

    @staticmethod
    def _dU(A: np.ndarray, U: np.ndarray, P: np.ndarray, dt) -> np.ndarray:
        """C.f. the tests for a naive impl."""
        return -np.sum(np.fmin(A * U[:, None], dt * P), axis=1) + np.sum(
            np.fmin(A.T * U[None, :], dt * P.T), axis=1
        )

    def update_state(self, U) -> np.ndarray:
        return np.fmin(
            (
                U
                + EnergySystem._dU(
                    self.weighted_A, U, self.process_power_limits, self.dt
                )
            )
            * (1 - self.U_frac_dt_loss),
            self.U_max,
        )

    def simulate(self):
        """Assumes AC from PV is only input and was the node initialising the graph."""
        u = np.array(self.g.node_apply(lambda n: n.value))
        us = np.zeros((HOURS_PER_YEAR, *u.shape))
        indep_vars = tuple(
            i for i, n in enumerate(self.g.nodes) if n.__dict__.get("outputs")
        )
        for i in range(HOURS_PER_YEAR):
            for j in indep_vars:
                u[j] = self.g.nodes[j].data.outputs["ac"][i] * WH
            us[i] = u
            u = self.update_state(u)
        self.us = us
        return us

    def plot(self, ax, exclude_cls_or_names=None):
        for i in range(self.us.shape[1]):
            if (
                exclude_cls_or_names is None
                or type(self.g.nodes[i]) not in exclude_cls_or_names
                and str(self.g.nodes[i]) not in exclude_cls_or_names
            ):
                ax.plot(self.us[:, i], label=str(self.g.inspect_ordering()[i]))
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Energy (J)")
        ax.legend()

    @property
    def total_useful(self):
        return self.us[-1, self.g.inspect_ordering().index("OUTPUT")]

    @property
    def total_input(self):
        return sum(
            self.us[:, i].sum()
            for i in range(self.us.shape[1])
            if self.g.nodes[i].__dict__.get("outputs")
        )

    @property
    def net_efficiency(self):
        return self.total_useful / self.total_input

    @property
    def total_lost(self):
        return self.us[-1, self.g.inspect_ordering().index("LOST")]


def basic_system(
    pv_spend, battery_spend, electrolyser_spend, h2storage_spend
) -> EnergySystem:
    out = EnergyStore(value=0, name="OUTPUT")
    sun = Sun(purchased=pv_spend)
    sun.link_to(
        Battery(purchased=battery_spend),
        using=PVProcessToBatteryConnection,
        purchased=pv_spend,
    ).node_to.link_to(
        H2Refueler(purchased=h2storage_spend),
        using=BatteryConnectionToElectrolyserCompressorProcess,
        purchased=electrolyser_spend,
    ).node_to.link_to(
        out
    )
    return EnergySystem(add_energy_sinks(sun.to_graph()))
