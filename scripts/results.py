import numpy as np
from pypvwatts.pypvwatts import PVWatts, PVWattsResult
import cydrogen
import cydrogen.optimise

# sample = PVWatts.request(
#     system_capacity=100,  # Nameplate capacity (kW)
#     module_type=1,  # Module type: 0: Standard, 1: Premium, 2: Thin film
#     array_type=1,  # Array type: 0: Fixed - Open Rack, 1: Fixed - Roof Mounted, 2: 1-Axis, 3: 1-Axis Backtracking, 4: 2-Axis
#     azimuth=190,  # Azimuth angle (degrees)
#     tilt=30,  # Tilt angle (degrees)
#     dataset="intl",  # climate dataset to use
#     losses=14,  # System losses (%)
#     lat=34.88,  # latitude for the location (north/south) - Larnaca
#     lon=33.63,  # longitude for the location (west/east)- Larnaca
#     timeframe="hourly",
# )


# def g(s, **kw):
#     _ = np.array(sample.raw["outputs"]["ac"]) / 100 * kw["system_capacity"]

#     class A:
#         ...

#     o = A()
#     o.raw = {"inputs": kw, "outputs": {"ac": _, "ac_annual": _.sum()}}
#     o.outputs = o.raw["outputs"]
#     o.inputs = o.raw["inputs"]
#     return o


# cydrogen.model.Sun._request = g


def f(num_hevs: int, x0=None):
    if x0 is None:
        x0 = [20e3, 20e3, 50e3, 100e3]
    opt = cydrogen.optimise.MinimiseHEVBudget(
        {
            "x0": np.array(x0) * num_hevs,  # np.ones(4) * 7e5,
            "bounds": [(3500, 3000e6)] + [(0, None)] * 3,  # bounds, PV has limits
            "options": {"disp": True},
            "method": "L-BFGS-B",
            "jac": "2-point",
        }
    )
    opt.output_limit = cydrogen.YEARLY_HYUNDAI_NEXO_ENERGY_CONSUMPTION * num_hevs
    return opt()


for n in [1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]:
    print("#############################################################")
    print(f"{n = }")
    print("---------------------------")
    x0 = None
    if n == 1:
        x0 = [7.500e03, 7.500e03, 1.875e04, 3.750e04]
        continue
    elif n == 1e1:
        x0 = [4.8e04, 4.3e04, 4.9e05, 2.345e05]
        continue
    elif n == 1e2:
        x0 = [4.7e05, 4.8e05, 1.3e06, 2.37e06]
        continue
    elif n == 1e3:
        x0 = [3e06, 806, 2e07, 4e07]
    elif n == 1e4:
        x0 = [7.500e07, 7.500e07, 1.875e08, 3.750e08]
    elif n == 1e5:
        x0 = [7.500e08, 7.500e08, 1.875e09, 3.750e09]
    elif n == 1e6:
        x0 = [7.500e09, 7.500e09, 1.875e10, 3.750e10]
    r = f(n, x0)
    print("Done optimising!!!")
    print(r)
    # print(f"For {n} HEVs, the total budget is {sum(r)} EUR.")
    print(f"The normalised budget ratio is {np.array(r.x) / sum(r.x)}")
    d = cydrogen.basic_system(*r.x)
    d.simulate()
    print(f"The efficiency of this system is {d.net_efficiency}.")
    print(
        f"The total kW of PV nameplate capacity is {d.g.nodes[0].inputs['system_capacity']}."
    )

# n=1
#   message: CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH
#   success: True
#    status: 0
#       fun: 70822.00000000492
#         x: [ 7.393e+03  7.393e+03  1.864e+04  3.739e+04]
#       nit: 4
#       jac: [ 1.000e+00  1.000e+00  1.000e+00  1.000e+00]
#      nfev: 189
#      njev: 21
#  hess_inv: <4x4 LbfgsInvHessProduct with dtype=float64>
# The normalised budget ratio is [0.10438847 0.10438847 0.26323741 0.52798565]
# The efficiency of this system is 0.5501126581722285.
# The total kW of PV nameplate capacity is 4.624210526316759.

# n = 10
# message: CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH
# success: True
# status: 0
#     fun: 714780.4711834376
#         x: [ 4.696e+04  4.174e+04  4.174e+05  2.087e+05]
#     nit: 4
#     jac: [ 1.000e+00  1.000e+00  1.000e+00  1.000e+00]
#     nfev: 369
#     njev: 41
# hess_inv: <4x4 LbfgsInvHessProduct with dtype=float64>
# The normalised budget ratio is [0.06569338 0.0583941  0.5839417  0.29197082]
# The efficiency of this system is 0.5501126581722328.
# The total kW of PV nameplate capacity is 46.26983458394116.

# n=100
# Not opt but f(439006.1191390952, 439006.1191390952, 1098185.853066367996, 2196818.619132974)=4173016.664047964
