"""
Utility functions for computational graph traversal and visualization.

This module provides helper utilities for:
    - tracing computational graph dependencies
    - extracting graph nodes and edges
    - visualizing autodiff graphs using Graphviz

Primarily used for debugging, graph inspection,
and understanding gradient propagation behavior.
"""

from graphviz import Digraph

def trace(root):
    """
    Traverse the computational graph rooted at `root`.

    This function performs a recursive graph traversal to collect all
    nodes and directed edges participating in the computational graph.

    The resulting graph representation is used for visualization,
    debugging, and inspection of forward computation dependencies.

    Graph semantics:
        - Nodes represent `Scalar` instances.
        - Directed edges represent parent-to-child data dependencies.

    Parameters
    ----------
    root : Scalar
        Output node representing the terminal value of a computation.

    Returns
    -------
    tuple[set, set]
        A tuple containing:

        - nodes:
            Set of all reachable graph nodes.

        - edges:
            Set of directed edges represented as:
                (parent_node, child_node)

    Notes
    -----
    Traversal is performed recursively using depth-first search (DFS).
    Duplicate visits are avoided through explicit node memoization.
    """

    nodes = set()
    edges = set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges

def draw_dot(root):
    """
    Generate a Graphviz visualization of the computational graph.

    This utility renders the dynamic computational graph associated
    with a scalar expression, including forward computation structure
    and gradient state information.

    Each graph node contains:
        - variable name
        - scalar value (`data`)
        - accumulated gradient (`grad`)

    Intermediate operation nodes are inserted between operand nodes
    and result nodes to explicitly represent differentiable operations.

    Parameters
    ----------
    root : Scalar
        Root node of the computational graph to visualize.

    Returns
    -------
    graphviz.Digraph
        Graphviz directed graph object representing the full
        computational graph.

    Visualization Details
    ---------------------
    - Data nodes represent scalar values.
    - Operation nodes represent differentiable operations.
    - Directed edges encode computational dependencies.
    - Graph layout is rendered left-to-right for readability.

    Notes
    -----
    This utility is primarily intended for:
        - debugging autodiff behavior
        - inspecting graph topology
        - educational visualization
        - validating gradient propagation paths
    """
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    nodes, edges = trace(root)

    for n in nodes:

        uid = str(id(n))

        # Data node
        dot.node(
            name=uid,
            label=f"{n.var_name} | data: {n.data:.4f} | grad: {n.grad:.4f}",
            shape='record'
        )

        # Operation node
        if n._op:
            op_id = uid + n._op
            dot.node(
                name=op_id,
                label=n._op
            )
            dot.edge(op_id, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    dot.render("computational_graph", view = True)
    return dot