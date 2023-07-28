"""
Olympios, Andreas: https://docs.google.com/document/d/15Eptzi4M0495eH9-LZqtBFhbYizCH8SR/edit#heading=h.1hmsyys
Alexandros, Arselis: https://doi.org/10.1016/j.renene.2018.03.060
H2 Storage: https://doi.org/10.1016/j.egypro.2012.09.076
"""
import dataclasses
import math
from typing import Optional, Tuple, Type
import numpy as np
from pypvwatts.pypvwatts import PVWatts, PVWattsResult
from .graph import Graph, Node, Edge, id_hash
from .units import EUR, H2_LHV, KILO, PERCENT, WH, HOUR, J, W, kWH, kW


PVWatts.api_key = "EF5VXVXX8NY0ONuhundhL5STynjc0vqhp9DsmKLz"
PVWatts.PVWATTS_QUERY_URL = "https://developer.nrel.gov/api/pvwatts/v8.json"

###############################################################################


def add_energy_sinks(g: Graph):
    s = EnergyStore(name="LOST")
    # for n in g.nodes:
    #     # make sure weights sum to 1
    #     if n.edges_out:
    #         if any(e.weight < 0 for e in n.edges_out):
    #             raise ValueError(
    #                 f"Negative edge weight is forbidden for: { {e for e in n.edges_out if e.weight < 0} }"
    #             )
    #         n.link_to(s, weight=1 - sum(e.weight for e in n.edges_out))
    for e in g.edges:
        # if hasattr(e, "efficiency"):
        e.node_from.link_to(s, weight=e.weight * (1 - e.efficiency))
        e.weight *= e.efficiency
    out_g = g.nodes[0].to_graph()
    assert g.nodes[0] is out_g.nodes[0]
    return out_g


###############################################################################


@dataclasses.dataclass(eq=False)
class EnergyStore(Node):
    purchased: float = 1e20  # np.inf in EUR
    specific_energy_cost: float = 1e-20  # ~0 in EUR / Wh
    """The rate of energy wastage."""
    static_frac_power_loss: float = 0
    installation_cost: float = 0
    max_energy_stored: float = dataclasses.field(init=False)

    def __post_init__(self):
        self.max_energy_stored = (
            self.purchased - self.installation_cost
        ) / self.specific_energy_cost

    def link_to(
        self,
        other: "Node",
        weight: float = 1,
        using: Optional[Type["Edge"]] = None,
        **edge_kw,
    ) -> "Edge":
        """This method overrides the superclass to allow for the use of a custom edge class.

        This preserves Liskov substitutability.
        """
        return super().link_to(other, weight, using or BasicEnergyProcess, **edge_kw)


@dataclasses.dataclass(eq=False)
class Battery(EnergyStore):
    specific_energy_cost: float = 500 * EUR / kWH
    static_frac_power_loss: float = 0.04 * PERCENT / HOUR
    installation_cost: float = 1500 * EUR


@dataclasses.dataclass(eq=False)
class H2Refueler(EnergyStore):
    specific_energy_cost: float = 10 * EUR / kWH
    static_frac_power_loss: float = 0
    installation_cost: float = 1500 * EUR


@id_hash
@dataclasses.dataclass(eq=False)
class Sun(EnergyStore):
    """

    Notes
    -----
    This is an extremely messy implementation of the Sun node.
    Ideally, the Sun should be a infinite power source,
    and have the PV constraints in the PV process itself.
    However, for the minimal demo API it was easier to do everything
    in this class.
    Nevertheless, it's theoretically quite easy to adjust;
    make sure to edit the cydrogen.system iterative model to account for this change.
    """

    data: PVWattsResult | None = None
    # NOTE: it is 950 EUR / kW in the paper, but that is not a unit of energy
    specific_energy_cost: float = 950 * EUR / kW / HOUR
    installation_cost: float = 3000 * EUR

    def __init__(self, purchased):
        super().__init__()

        # idk why this is needed; dataclass inheritance is pretty buggy
        self.specific_energy_cost: float = 950 * EUR / kW / HOUR
        self.installation_cost: float = 3000 * EUR

        # PVWatts.request returns a dictionary with the following keys: 'inputs', 'errors', 'warnings', 'version', 'ssc_info', 'station_info', 'location', 'outputs', 'temperature_air', 'temperature_cell', 'wind_speed'
        self.data = PVWatts.request(
            system_capacity=(purchased - self.installation_cost)
            / (self.specific_energy_cost * KILO * HOUR),  # Nameplate capacity (kW)
            module_type=1,  # Module type: 0: Standard, 1: Premium, 2: Thin film
            array_type=1,  # Array type: 0: Fixed - Open Rack, 1: Fixed - Roof Mounted, 2: 1-Axis, 3: 1-Axis Backtracking, 4: 2-Axis
            azimuth=190,  # Azimuth angle (degrees)
            tilt=30,  # Tilt angle (degrees)
            dataset="intl",  # climate dataset to use
            losses=14,  # System losses (%)
            lat=34.88,  # latitude for the location (north/south) - Larnaca
            lon=33.63,  # longitude for the location (west/east)- Larnaca
            timeframe="hourly",
        )
        self.inputs = self.data.inputs
        self.outputs = self.data.raw["outputs"]


###############################################################################


class BasicEnergyProcess(Edge):
    purchased: float  # np.inf in EUR
    max_power_transfer: float
    specific_power_cost: float = 1e-20  # ~0 in EUR / W
    """The maximum amount of energy that can be transferred through the edge in a single second."""
    efficiency: float = 1
    installation_cost: float = 0

    def __init__(self, node_from: Node, node_to: Node, weight=1, purchased=1e20):
        super().__init__(node_from, node_to, weight)
        self.purchased = purchased
        self.max_power_transfer = (
            purchased - self.installation_cost
        ) / self.specific_power_cost


class PVProcess(BasicEnergyProcess):
    specific_power_cost: float = 1e-20  # see Sun for price impl.
    efficiency: float = 1  # accounted for already by PVWatts
    installation_cost: float = 0  # accounted for in Sun class


class ElectrolyserCompressorProcess(BasicEnergyProcess):
    specific_power_cost: float = 430 * EUR / kW
    efficiency: float = 0.75
    installation_cost: float = 2200 * EUR


class BatteryConnection(BasicEnergyProcess):
    # round-trip is 0.88 (Olympios 2023), so single trip is sqrt(0.88)
    efficiency: float = 0.88**0.5
    installation_cost: float = 1500 * EUR


class PVProcessToBatteryConnection(BasicEnergyProcess):
    specific_power_cost: float = PVProcess.specific_power_cost
    efficiency: float = BatteryConnection.efficiency
    installation_cost: float = (
        PVProcess.installation_cost + BatteryConnection.installation_cost
    )


class BatteryConnectionToElectrolyserCompressorProcess(BasicEnergyProcess):
    specific_power_cost: float = ElectrolyserCompressorProcess.specific_power_cost
    efficiency: float = (
        ElectrolyserCompressorProcess.efficiency * BatteryConnection.efficiency
    )
    installation_cost: float = (
        BatteryConnection.installation_cost
        + ElectrolyserCompressorProcess.installation_cost
    )
