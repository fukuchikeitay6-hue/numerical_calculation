import numpy as np

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Material:
    def __init__(self, k):
        self.k = k  #拡散係数

class Element:
    def __init__(self, node_indexes, material) -> None:
        self.node_indexes = np.array(node_indexes)
        self.material = material
        self.Ke = None
        self.fe = None

class Mesh:
    def __init__(self, nodes, elements) -> None:
        self.nodes = nodes
        self.element = elements
        self.num_nodes = len(nodes)
        self.num_dofs = self.num_nodes

class BoundaryCondition:
    def __init__(self, num_dofs) -> None:
        self.dirichlet_bcs = {}
        self.neumann_bcs = np.zeros(num_dofs)

        def set_dirichlet_bc(self, node_index, value):
            self.dirichlet_bcs[node_index] = value

        def add_neumann_bc(self, node_index, flux_value):
            self.neumann_bcs[node_index] += flux_value
