from typing import Any, Callable, Optional, Set, Type
import dataclasses
from ordered_set import OrderedSet
import numpy as np


def id_hash(obj: object) -> int:
    obj.__hash__ = lambda self: id(self)
    return obj


class Edge:
    node_from: "Node"
    node_to: "Node"
    weight: float = 1

    def __init__(self, node_from: "Node", node_to: "Node", weight=1):
        for e in node_from.edges_out:
            if e.node_to is node_to:
                raise ValueError(
                    "Cannot create duplicate edge. Use a dummy node as an intermediary."
                )
        self.node_from = node_from
        self.node_to = node_to
        self.weight = weight
        node_from.edges_out.add(self)
        node_to.edges_in.add(self)

    def __hash__(self) -> int:
        return hash((self.node_from, self.node_to, self.weight))

    def __eq__(self, other: "Edge") -> bool:
        return (self.node_from, self.node_to, self.weight) == (
            other.node_from,
            other.node_to,
            other.weight,
        )

    @property
    def name(self):
        return self.__class__.__name__

    def __str__(self):
        return self.name or self.__class__.__name__


@id_hash
@dataclasses.dataclass
class Node:
    value: float = 0
    name: Optional[str] = None
    edges_in: Set["Edge"] = dataclasses.field(default_factory=set)
    edges_out: Set["Edge"] = dataclasses.field(default_factory=set)
    data: dict = dataclasses.field(default_factory=dict)

    def to_graph(self) -> "Graph":
        nodes: OrderedSet[Node] = OrderedSet()
        edges: Set[Node] = set()
        stack = [self]
        while stack:
            node = stack.pop()
            nodes.add(node)
            stack.extend(
                edge.node_from for edge in node.edges_in if edge.node_from not in nodes
            )
            stack.extend(
                edge.node_to for edge in node.edges_out if edge.node_to not in nodes
            )
            edges |= node.edges_in | node.edges_out
        return Graph(nodes, edges)

    def link_to(
        self,
        other: "Node",
        weight: float = 1,
        using: Type[Edge] | None = None,
        **edge_kw,
    ) -> "Edge":
        # if self is other:
        #     raise ValueError(
        #         "Cannot link a node to itself; use a dummy node as an intermediary."
        #     )
        for e in self.edges_out:
            if e.node_to is other:
                if e.__class__ is not using:
                    raise ValueError(
                        f"Cannot link {self} to {other} using {using or Edge} because it is already linked using {e}."
                    )
                e.weight += weight
                return e
        return (using or Edge)(self, other, weight, **edge_kw)

    def __str__(self):
        return self.name or self.__class__.__name__


class Graph:
    nodes: OrderedSet[Node]
    edges: Set[Edge]

    def __init__(self, nodes=None, edges=None):
        self.nodes = OrderedSet(nodes or [])
        self.edges = set(edges or [])

    def add_node(self, node: Node):
        self.nodes.add(node)
        return node

    def add_edge(self, edge: Edge):
        self.edges.add(edge)
        self.nodes |= {edge.node_from, edge.node_to}
        return edge

    @property
    def cardinality(self):
        return len(self.nodes)

    def get_adjacency_matrix(self, edge_fn: Callable[[Edge], Any] = bool) -> np.ndarray:
        """Return a numpy array representing the adjacency matrix of the graph."""
        arr = np.zeros((self.cardinality, self.cardinality))
        for n, node in enumerate(self.nodes):
            for edge in node.edges_out:
                arr[n, self.nodes.index(edge.node_to)] = edge_fn(edge)
        return arr

    def node_apply(self, fn: Callable[[Node], Any]) -> None:
        return [fn(node) for node in self.nodes]

    def __repr__(self):
        return "\n".join(
            f"{e.node_from} to {e.node_to} linked by {e.__class__.__name__} where {e.weight=}"
            for e in self.edges
        )

    def inspect_ordering(self):
        return [str(n) for n in self.nodes]

    def to_nx(self):
        import networkx as nx

        g = nx.DiGraph()
        g.add_weighted_edges_from(
            (str(e.node_from), str(e.node_to), e.weight) for e in self.edges
        )
        return g

    def nx_draw(self, edge_attrs=None):
        import networkx as nx

        G = self.to_nx()
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True)
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels={
                (str(e.node_from), str(e.node_to)): str(e) for e in self.edges
            },
        )
        print(G.edges(data=True))
        if edge_attrs:
            for a in edge_attrs:
                nx.draw_networkx_edge_labels(
                    G, pos, edge_labels={(u, v): d[a] for u, v, d in G.edges(data=True)}
                )

    def nx_show(self, edge_attrs=None):
        import matplotlib.pyplot as plt

        self.nx_draw(edge_attrs)
        plt.show()

    @classmethod
    def from_adjacency_matrix(cls, A: np.ndarray[Any, bool]):
        """Construct a graph from an adjacency matrix."""
        g = cls()
        for i in range(A.shape[0]):
            g.add_node(Node(name=i))
        for (i, j), val in np.ndenumerate(A):
            if val:
                g.add_edge(Edge(g.nodes[i], g.nodes[j], weight=val))
        return g

    @classmethod
    def Kn(cls, n: int):
        return cls.from_adjacency_matrix(1 - np.eye(n))
