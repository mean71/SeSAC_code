import sys 
sys.path.append('../data_structure')

try:
    from graph_datastructure import AdjList, AdjMatrix, Vertex, Edge
except ModuleNotFoundError:
    from data_structure.graph_datastructure import AdjList, AdjMatrix, Vertex, Edge


class Graph:
    def __init__(self, V, E, backend = 'VE'):
        for v in V:
            assert isinstance(v, Vertex) 
        for e in E:
            assert isinstance(e, Edge)
            assert e.from_vertex in V 
            assert e.to_vertex in V 

        self.V = V 
        self.E = E 

        if backend == 'VE':
            pass 
        elif backend == 'adjacent_list':
            pass 
        elif backend == 'adjacnet_matrix':
            pass 

    def add_vertex(self, v):
        assert isinstance(v, Vertex)
    
    def remove_vertex(self, v):
        assert isinstance(v, Vertex)

    def add_edge(self, e):
        assert isinstance(e, Edge)

    def remove_edge(self, e):
        assert isinstance(e, Edge)

    def get_vertices(self):
        return [] 

    def get_neighbors(self, v):
        assert isinstance(v, Vertex)
        return [] 

    def dfs(self, src):
        assert isinstance(src, Vertex) 
        yield None 

    def bfs(self, src):
        assert isinstance(src, Vertex) 
        yield None 


    # Do not modify this method

    @staticmethod
    def spring_layout(nodes, edges, iterations=50, k=0.1, repulsion=0.01):
        import numpy as np
        # Initialize positions randomly
        positions = {node: np.random.rand(2) for node in nodes}
        
        for _ in range(iterations):
            forces = {node: np.zeros(2) for node in nodes}
            
            # Repulsive forces between all pairs of nodes
            for i, node1 in enumerate(nodes):
                for j, node2 in enumerate(nodes):
                    if i != j:
                        diff = positions[node1] - positions[node2]
                        dist = np.linalg.norm(diff)
                        if dist > 0:
                            forces[node1] += (diff / dist) * repulsion / dist**2
            
            # Attractive forces for connected nodes
            for edge in edges:
                node1, node2 = edge.from_vertex, edge.to_vertex
                diff = positions[node2] - positions[node1]
                dist = np.linalg.norm(diff)
                
                if dist > 0:
                    force = k * (dist - 1)  # spring force
                    forces[node1] += force * (diff / dist)
                    forces[node2] -= force * (diff / dist)
            
            # Update positions
            for node in nodes:
                positions[node] += forces[node]
        
        return positions

    def show(self):
        import matplotlib.pyplot as plt
        nodes = self.V 
        edges = self.E 
        positions = Graph.spring_layout(nodes, edges)
        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        # Plot nodes
        for node, pos in positions.items():
            ax.scatter(*pos, s=2000, color='lightblue')
            ax.text(*pos, node, fontsize=20, ha='center', va='center')

        # Plot edges
        for edge in edges:
            node1, node2 = edge.from_vertex, edge.to_vertex
            x_values = [positions[node1][0], positions[node2][0]]
            y_values = [positions[node1][1], positions[node2][1]]
            ax.plot(x_values, y_values, color='gray', linewidth=2)

        ax.set_title("Graph Visualization with Spring Layout", fontsize=20)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()


if __name__ == '__main__':
    v1 = Vertex(0, 1)
    v2 = Vertex(1, 2)
    v3 = Vertex(2, 3)
    v4 = Vertex(3, 4)
    v5 = Vertex(4, 5)

    e1 = Edge(v1, v2) 
    e2 = Edge(v1, v3) 
    e3 = Edge(v2, v3)
    e4 = Edge(v2, v4)
    e5 = Edge(v3, v5) 
    e6 = Edge(v4, v5)

    V = [v1, v2]
    E = [e1]

    g1 = Graph(V, E) 

    g1.add_vertex(v3)
    g1.add_vertex(v4)
    g1.add_vertex(v5)

    g1.add_edge(e2)
    g1.add_edge(e3)
    g1.add_edge(e4)
    g1.add_edge(e5)
    g1.add_edge(e6)

    g1.show()



