from cydrogen.graph import Graph, Node, Edge


def test_Graph():
    a = Node(1)
    b = Node(2)
    ab = Edge(a, b)
    ba = Edge(b, a, 2)
    aa = a.link_to(a, 3)
    g = a.to_graph()
    assert g.nodes == {a, b}
    assert g.edges == {ab, ba, aa}
    assert g.cardinality == 2
    assert g.nodes[0] == a
    assert g.nodes[1] == b
    assert (g.get_adjacency_matrix().astype(int) == [[1, 1], [1, 0]]).all()
    assert (
        g.get_adjacency_matrix(lambda e: e.weight).astype(int) == [[3, 1], [2, 0]]
    ).all()


def test_factories():
    g = Graph.Kn(5)
    G = g.nodes[0].to_graph()
    assert g.nodes[0] is G.nodes[0]
    assert g.edges == G.edges
    assert g.cardinality == G.cardinality == 5
    assert g.nodes - G.nodes == G.nodes - g.nodes == set()
