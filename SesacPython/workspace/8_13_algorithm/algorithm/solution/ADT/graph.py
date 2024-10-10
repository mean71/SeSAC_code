import sys 
import os 

cur_path = os.path.abspath(__file__)
sys.path.append(f'{cur_path}/../data_structure')
sys.path.append(f'{cur_path}/..')

from data_structure.graph import AdjList, AdjMatrix, Vertex, Edge
from ADT.queue import Queue 
from ADT.stack import Stack 
    

class Graph:
    """
    Represents a graph data structure.

    Attributes:
    - backend (str): The backend representation ('VE'). Should be one of 'VE', 'adjacent_list', 'adjacent_matrix'. 

    Detailed Explanation:
    The Graph class represents a graph using a vertex list and an edge list ('VE' backend). This allows for flexibility in representing complex graphs, including cycles and multiple connections.

    Practical Usages:
    Graphs are fundamental in computer science and are used in networking, social networks, transportation systems, and more.
    """
    def __init__(self, V, E, backend = 'VE'):
        """
        Initializes a new Graph instance.

        Parameters:
        - V (list): A list of Vertex instances.
        - E (list): A list of Edge instances.
        - backend (str, optional): The backend representation. Defaults to 'VE'.

        Raises:
        - AssertionError: If V contains non-Vertex instances or E contains non-Edge instances.
        - ValueError: If an edge connects vertices not in V.

        Example:
            # Create vertices
            vA = Vertex(0, 'A')
            vB = Vertex(1, 'B')
            vC = Vertex(2, 'C')
            vD = Vertex(3, 'D')
            vE = Vertex(4, 'E')
            vF = Vertex(5, 'F')
            vG = Vertex(6, 'G')

            # Create edges
            eAB = Edge(vA, vB)
            eAC = Edge(vA, vC)
            eBD = Edge(vB, vD)
            eCD = Edge(vC, vD)
            eDE = Edge(vD, vE)
            eEF = Edge(vE, vF)
            eFG = Edge(vF, vG)
            eGE = Edge(vG, vE)  # Creates a cycle

            # Initialize graph
            V = [vA, vB, vC, vD, vE, vF, vG]
            E = [eAB, eAC, eBD, eCD, eDE, eEF, eFG, eGE]
            g = Graph(V, E)

            # The graph now contains 7 vertices and 8 edges, with cycles.
        """
        for v in V:
            assert isinstance(v, Vertex) 
        for e in E:
            assert isinstance(e, Edge)
            assert e.from_vertex in V 
            assert e.to_vertex in V 

        self.V = V 
        self.E = E
        self.backend = backend  

        if self.backend == 'VE':
            pass 
        elif self.backend == 'adjacent_list':
            self.adj_list = AdjList(V, E)
        elif self.backend == 'adjacnet_matrix':
            self.adj_matrix = AdjMatrix(V, E)
        else:
            raise ValueError('Invalid Backend')

    def add_vertex(self, v):
        """
        Adds a vertex to the graph.

        Parameters:
        - v (Vertex): The vertex to add.

        Raises:
        - ValueError: If the vertex already exists.

        Example:
            # Adding a new vertex 'H' to the graph
            vH = Vertex(7, 'H')
            g.add_vertex(vH)

            # Now, the graph contains vertices 'A' to 'H'.

        ASCII-Art Illustration:
            Before adding 'H':
                Graph structure (simplified):

                    A
                   / \
                  B   C
                   \ /
                    D
                    |
                    E
                   / \
                  F---G

            After adding 'H' (not connected yet):

                    A
                   / \
                  B   C
                   \ /
                    D
                    |
                    E
                   / \
                  F---G

                    H (isolated)
        """
        assert isinstance(v, Vertex)
        if self.backend == 'VE':
            if v not in self.V:
                self.V.append(v)
            else:
                raise ValueError(f'{v} is already in the graph')
        elif self.backend == 'adjacent_list':
            self.adj_list.add_vertex(v)
        elif self.backend == 'adjacnet_matrix':
            self.adj_matrix.add_vertex(v)
    
    def remove_vertex(self, v):
        """
        Removes a vertex from the graph.

        Parameters:
        - v (Vertex): The vertex to remove.

        Raises:
        - ValueError: If the vertex is not in the graph.

        Example:
            # Removing vertex 'C' from the graph
            g.remove_vertex(vC)

            # The graph now only contains vertices 'A', 'B', 'D', 'E', 'F', 'G'.

        ASCII-Art Illustration:
            Before removing 'C':

                    A
                   / \
                  B   C
                   \ /
                    D
                    |
                    E
                   / \
                  F---G

            After removing 'C':

                    A
                    |
                    B
                    |
                    D
                    |
                    E
                   / \
                  F---G

            'C' and its connecting edges are removed.
        """
        assert isinstance(v, Vertex)
        if self.backend == 'VE':
            try:
                self.V.remove(v)
            except ValueError as e:
                raise ValueError(f'{v} not in graph')
        elif self.backend == 'adjacent_list':
            self.adj_list.remove_vertex(v)
        elif self.backend == 'adjacnet_matrix':
            self.adj_matrix.remove_vertex(v)

    def add_edge(self, e):
        """
        Adds an edge to the graph.

        Parameters:
        - e (Edge): The edge to add.

        Raises:
        - AssertionError: If the edge connects vertices not in the graph.

        Example:
            # Adding an edge between 'H' and 'E'
            eHE = Edge(vH, vE)
            g.add_edge(eHE)

            # Now, 'H' is connected to 'E'.

        ASCII-Art Illustration:
            Before adding edge (H, E):

                    A
                   / \
                  B   C
                   \ /
                    D
                    |
                    E
                   / \
                  F---G

                    H

            After adding edge (H, E):

                    A
                   / \
                  B   C
                   \ /
                    D
                    |
                    E
                   /|\
                  F G H
        """
        assert isinstance(e, Edge)
        assert e.from_vertex in self.get_vertices()
        assert e.to_vertex in self.get_vertices()
        
        if self.backend == 'VE':
            self.E.append(e) 
        elif self.backend == 'adjacent_list':
            self.adj_list.add_edge(e)
        elif self.backend == 'adjacnet_matrix':
            self.adj_matrix.add_edge(e)

    def remove_edge(self, e):
        """
        Removes an edge from the graph.

        Parameters:
        - e (Edge): The edge to remove.

        Raises:
        - ValueError: If the edge is not in the graph.

        Example:
            # Removing the edge between 'E' and 'F'
            g.remove_edge(eEF)

            # Now, 'E' and 'F' are no longer directly connected.

        ASCII-Art Illustration:
            Before removing edge (E, F):

                    A
                   / \
                  B   C
                   \ /
                    D
                    |
                    E
                   /|\
                  F G H

            After removing edge (E, F):

                    A
                   / \
                  B   C
                   \ /
                    D
                    |
                    E
                   / \
                  G   H

                F (disconnected from 'E', but may still be connected through other paths)
        """
        assert isinstance(e, Edge)
        if e not in self.get_edges():
            raise ValueError(f'{e} not in graph')
        assert e.from_vertex in self.get_vertices()
        assert e.to_vertex in self.get_vertices()
        if self.backend == 'VE':
            self.E.remove(e)
        elif self.backend == 'adjacent_list':
            self.adj_list.remove_edge(e)
        elif self.backend == 'adjacnet_matrix':
            self.adj_matrix.remove_edge(e)

    def get_vertices(self):
        """
        Returns the list of vertices in the graph.

        Returns:
        - list: A list of Vertex instances.

        Example:
            vertices = g.get_vertices()
            print([str(v) for v in vertices])
            # Output: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        """
        if self.backend == 'VE':
            return self.V 
        elif self.backend == 'adjacent_list':
            return self.adj_list.get_vertices()
        elif self.backend == 'adjacnet_matrix':
            return self.adj_matrix.get_vertices()
         
    def get_edges(self):
        """
        Returns the list of edges in the graph.

        Returns:
        - list: A list of Edge instances.

        Example:
            edges = g.get_edges()
            print([(str(e.from_vertex), str(e.to_vertex)) for e in edges])
            # Output: [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D'), ('D', 'E'), ('E', 'F'), ('F', 'G'), ('G', 'E'), ('H', 'E')]
        """
        if self.backend == 'VE':
            return self.E 
        elif self.backend == 'adjacent_list':
            return self.adj_list.get_edges()
        elif self.backend == 'adjacnet_matrix':
            return self.adj_matrix.get_edges()

    def get_neighbors(self, v):
        """
        Returns the neighbors of a given vertex.

        Parameters:
        - v (Vertex): The vertex for which to find neighbors.

        Returns:
        - list: A list of neighboring Vertex instances.

        Example:
            neighbors = g.get_neighbors(vE)
            print([str(n) for n in neighbors])  # Output: ['D', 'F', 'G', 'H']

        ASCII-Art Illustration:
            Graph:

                    D
                    |
                    E
                   /|\
                  F G H

            Neighbors of 'E' are 'D', 'F', 'G', and 'H'.
        """
        assert isinstance(v, Vertex)
        if self.backend == 'VE':
            res = []
            
            for e in self.get_edges():
                if e.from_vertex == v:
                    res.append(e.to_vertex)
                if not e.is_directed:
                    if e.to_vertex == v:
                        res.append(e.from_vertex)
            
            return res 
        elif self.backend == 'adjacent_list':
            return self.adj_list.get_neighbors(v)
        elif self.backend == 'adjacnet_matrix':
            return self.adj_matrix.get_neighbors(v)

    def dfs(self, src):
        """
        Performs a depth-first search starting from the given vertex.

        Parameters:
        - src (Vertex): The starting vertex.

        Returns:
        - list: A list of visited Vertex instances in the order they were visited.

        Example:
            dfs_result = g.dfs(vA)
            print([str(v) for v in dfs_result])
            # Possible Output: ['A', 'C', 'D', 'E', 'H', 'G', 'F', 'B']

        ASCII-Art Illustration:
            Graph:

                    A
                   / \
                  B   C
                   \ /
                    D
                    |
                    E
                   /|\
                  F G H

            Traversal Order:
            - Start at 'A'.
            - Visit 'C' (neighbor of 'A').
            - From 'C', visit 'D'.
            - From 'D', visit 'E'.
            - From 'E', visit 'H'.
            - Backtrack to 'E', visit 'G'.
            - Backtrack to 'E', visit 'F'.
            - Backtrack to 'E', then 'D', then 'C', then 'A'.
            - Visit 'B' (remaining neighbor of 'A').
        """
        assert isinstance(src, Vertex) 
        
        if self.backend == 'VE':
            s = Stack(src)
            visited = []

            while not s.is_empty():
                cur = s.pop()
                visited.append(cur)
                neighbors = self.get_neighbors(cur)
                neighbors.sort(key = lambda x:x.datum, reverse = True)
                for n in neighbors:
                    if n not in visited and n not in s.elements():
                        s.push(n)
            
            return visited 

        elif self.backend == 'adjacent_list':
            pass 
        elif self.backend == 'adjacnet_matrix':
            pass 
        
    def bfs(self, src):
        """
        Performs a breadth-first search starting from the given vertex.

        Parameters:
        - src (Vertex): The starting vertex.

        Returns:
        - list: A list of visited Vertex instances in the order they were visited.

        Example:
            bfs_result = g.bfs(vA)
            print([str(v) for v in bfs_result])
            # Output: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

        ASCII-Art Illustration:
            Graph:

                    A
                   / \
                  B   C
                   \ /
                    D
                    |
                    E
                   /|\
                  F G H

            Traversal Order:
            - Start at 'A'.
            - Visit immediate neighbors: 'B' and 'C'.
            - Next, visit neighbors of 'B' and 'C': 'D'.
            - Visit neighbors of 'D': 'E'.
            - Visit neighbors of 'E': 'F', 'G', 'H'.
        """
        assert isinstance(src, Vertex) 
        if self.backend == 'VE':
            s = Queue(src)
            visited = []

            while not s.is_empty():
                cur = s.dequeue()
                visited.append(cur)
                neighbors = self.get_neighbors(cur)
                neighbors.sort(key = lambda x:x.datum, reverse = True)
                for n in neighbors:
                    if n not in visited and n not in s.elements():
                        s.enqueue(n)
            
            return visited 
        elif self.backend == 'adjacent_list':
            pass 
        elif self.backend == 'adjacnet_matrix':
            pass 
        
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

Edge = Edge 
Vertex = Vertex 

if __name__ == '__main__':
    q = Queue([1,2,3])
    s = Stack([1,2,3])
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

    
    for n in g1.bfs(v1):
        print(n)
    # g1.show()    

    vA = Vertex(0, 'A')
    vB = Vertex(1, 'B')
    vC = Vertex(2, 'C')
    vD = Vertex(3, 'D')
    vE = Vertex(4, 'E')
    vF = Vertex(5, 'F')
    vG = Vertex(6, 'G')

    # Create edges
    eAB = Edge(vA, vB)  # Edge A-B
    eAC = Edge(vA, vC)  # Edge A-C
    eBD = Edge(vB, vD)  # Edge B-D
    eCD = Edge(vC, vD)  # Edge C-D
    eDE = Edge(vD, vE)  # Edge D-E
    eEF = Edge(vE, vF)  # Edge E-F
    eFG = Edge(vF, vG)  # Edge F-G
    eGE = Edge(vG, vE)  # Edge G-E (creates a cycle)

    # Initialize graph
    V = [vA, vB, vC, vD, vE, vF, vG]
    E = [eAB, eAC, eBD, eCD, eDE, eEF, eFG, eGE]
    g = Graph(V, E, backend='VE')

    # Initial Ascii-Art Representation
    print("Initial Graph:")
    print("""
        A
       / \\
      B   C
       \\ /
        D
        |
        E
       / \\
      F---G
    """)

    # 1. Adding a Vertex 'H'
    print("Adding Vertex 'H':")
    vH = Vertex(7, 'H')
    g.add_vertex(vH)
    print("Vertices after adding 'H':", [str(v) for v in g.get_vertices()])

    # Ascii-Art (Vertex 'H' is not connected yet)
    print("""
        A
       / \\
      B   C
       \\ /
        D
        |
        E
       / \\
      F---G

       H (isolated)
    """)

    # 2. Adding an Edge between 'H' and 'E'
    print("Adding Edge between 'H' and 'E':")
    eHE = Edge(vH, vE)  # Edge H-E
    g.add_edge(eHE)
    print("Edges after adding (H, E):", [(str(e.from_vertex), str(e.to_vertex)) for e in g.get_edges()])

    # Ascii-Art after adding edge (H, E)
    print("""
        A
       / \\
      B   C
       \\ /
        D
        |
        E
       /|\\
      F G H
    """)

    # 3. Removing an Edge between 'G' and 'E'
    print("Removing Edge between 'G' and 'E':")
    g.remove_edge(eGE)  # Remove Edge G-E
    print("Edges after removing (G, E):", [(str(e.from_vertex), str(e.to_vertex)) for e in g.get_edges()])

    # Ascii-Art after removing edge (G, E)
    print("""
        A
       / \\
      B   C
       \\ /
        D
        |
        E
       /|\\
      F G H

    (Edge between G and E is removed)
    """)

    # 4. Removing a Vertex 'C'
    print("Removing Vertex 'C':")
    g.remove_vertex(vC)  # Remove Vertex C
    print("Vertices after removing 'C':", [str(v) for v in g.get_vertices()])
    print("Edges after removing 'C':", [(str(e.from_vertex), str(e.to_vertex)) for e in g.get_edges()])

    # Ascii-Art after removing Vertex 'C'
    print("""
        A
        |
        B
        |
        D
        |
        E
       /|\\
      F G H

    (Vertex 'C' and its edges are removed)
    """)

    # 5. Getting Neighbors of 'E'
    print("Neighbors of Vertex 'E':")
    neighbors_of_E = g.get_neighbors(vE)
    print([str(n) for n in neighbors_of_E])  # Output: ['D', 'F', 'H']

    # 6. Depth-First Search (DFS) starting from 'A'
    print("DFS starting from Vertex 'A':")
    dfs_result = g.dfs(vA)
    print([str(v) for v in dfs_result])  # Possible Output: ['A', 'B', 'D', 'E', 'H', 'G', 'F']

    # 7. Breadth-First Search (BFS) starting from 'A'
    print("BFS starting from Vertex 'A':")
    bfs_result = g.bfs(vA)
    print([str(v) for v in bfs_result])  # Output: ['A', 'B', 'D', 'E', 'F', 'G', 'H']

    # 8. Performing BFS starting from 'E'
    print("BFS starting from Vertex 'E':")
    bfs_result_E = g.bfs(vE)
    print([str(v) for v in bfs_result_E])  # Output: ['E', 'D', 'F', 'G', 'H', 'B', 'A']

    # 9. Performing DFS starting from 'E'
    print("DFS starting from Vertex 'E':")
    dfs_result_E = g.dfs(vE)
    print([str(v) for v in dfs_result_E])  # Possible Output: ['E', 'H', 'G', 'F', 'D', 'B', 'A']




