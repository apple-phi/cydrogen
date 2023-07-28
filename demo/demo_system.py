import matplotlib.pyplot as plt
import cydrogen

s_optimised = cydrogen.basic_system(
    645223.4719163576,
    243960.66318646292,
    106915.50432184557,
    3900.360575333828,
)
s_naive = cydrogen.basic_system(*[2.5e5] * 4)
s_optimised.simulate()
ax_indiv = plt.figure("Individual").add_subplot()
s_optimised.plot(ax_indiv, exclude_cls_or_names=(cydrogen.Sun,))
s_naive.simulate()
ax_cmp = plt.figure("Comparison").add_subplot()
ax_cmp.plot(
    s_optimised.us[:, s_optimised.g.inspect_ordering().index("OUTPUT")],
    label="Optimised",
)
ax_cmp.plot(s_naive.us[:, s_naive.g.inspect_ordering().index("OUTPUT")], label="Naive")
ax_cmp.legend()
ax_cmp.set_xlabel("Time (hours)")
ax_cmp.set_ylabel("Energy (J)")

print("Comparison of optimised and naive systems given a budget of 1e6 EUR:")
print()
print("Naive system")
print("------------")
print(f"Useful output: {s_naive.total_useful}")
print(f"Total input: {s_naive.total_input}")
print(f"Net efficiency: {s_naive.net_efficiency}")
assert s_naive.g.nodes[0].__class__ is cydrogen.Sun
print(f"kW of PVs used: {s_naive.g.nodes[0].inputs['system_capacity']}")
print()
print("Optimised system")
print("----------------")
print(f"Useful output: {s_optimised.total_useful}")
print(f"Total input: {s_optimised.total_input}")
print(f"Net efficiency: {s_optimised.net_efficiency}")
assert s_optimised.g.nodes[0].__class__ is cydrogen.Sun
print(f"kW of PVs used: {s_optimised.g.nodes[0].inputs['system_capacity']}")

plt.show()
