class Vertex:
    """
    Represents a vertex in the graph.

    Attributes:
    - id (Any): A unique identifier for the vertex.
    - datum (Any): The data stored in the vertex.

    Detailed Explanation:
    The Vertex class represents a node in the graph. Each vertex has an identifier and can store additional data.

    Practical Usages:
    Vertices are used in graph data structures to represent entities such as cities in a map, users in a social network, or web pages in a hyperlink structure.
    """
    def __init__(self, node_id, datum):
        self.datum = datum 
        self.node_id = node_id 

    def __eq__(self, other):
        if isinstance(self, Vertex) and isinstance(other, Vertex):
            return self.node_id == other.node_id
        return False 

    def __hash__(self):
        return hash((self.node_id, self.datum))

    def __str__(self):
        return str(self.datum)

class Edge:
    """
    Represents an edge in the graph.

    Attributes:
    - from_vertex (Vertex): The starting vertex of the edge.
    - to_vertex (Vertex): The ending vertex of the edge.
    - is_directed (bool): Indicates whether the edge is directed.

    Detailed Explanation:
    The Edge class connects two vertices in the graph. If the edge is directed, it goes from 'from_vertex' to 'to_vertex'.

    Practical Usages:
    Edges are used to represent relationships or connections between entities in a graph, such as roads between cities, friendships between people, or hyperlinks between web pages.
    """
    def __init__(self, from_vertex, to_vertex, is_directed = True, **data):
        assert isinstance(from_vertex, Vertex)    
        self.from_vertex = from_vertex

        assert isinstance(to_vertex, Vertex)
        self.to_vertex = to_vertex

        self.is_directed = is_directed
        self.data = data 
    
    def __eq__(self, other):
        if isinstance(self, Edge) and isinstance(other, Edge):
            return self.from_vertex == other.from_vertex and self.to_vertex == other.to_vertex


class AdjList:
    """
    Represents an adjacency list for graph representation.

    Attributes:
    - adj_list (dict): A dictionary mapping each vertex to a list of its neighboring vertices.

    Detailed Explanation:
    An adjacency list represents a graph by maintaining a list of adjacent vertices for each vertex. It's efficient for sparse graphs and allows for quick lookup of neighbors.

    Practical Usages:
    Adjacency lists are commonly used in graph algorithms where space efficiency is important, such as representing social networks or the World Wide Web.

    """
    def __init__(self, V, E):
        """
        Initializes the adjacency list representation of a graph.

        Parameters:
        - V (list of Vertex): A list of Vertex instances representing the vertices of the graph.
        - E (list of Edge): A list of Edge instances representing the edges of the graph.

        Returns:
        - None

        Detailed Explanation:
        This method initializes the adjacency list by creating a dictionary where each vertex maps to a list of its neighboring vertices. It populates the adjacency list based on the provided edges, considering whether the edges are directed or undirected.

        Implementation Steps:
        1. Create an empty adjacency list for each vertex in V.
        2. Iterate over each edge in E:
           - Add the 'to_vertex' to the adjacency list of 'from_vertex'.
           - If the edge is undirected, also add the 'from_vertex' to the adjacency list of 'to_vertex'.

        Example:
            v1 = Vertex(1, 'A')
            v2 = Vertex(2, 'B')
            e1 = Edge(v1, v2)
            adj_list = AdjList([v1, v2], [e1])
            # adj_list.adj_list will be {v1: [v2], v2: []} for a directed edge.
        """
        self.adj_list = {v: [] for v in V}
        for e in E:
            self.adj_list[e.from_vertex].append(e.to_vertex)
            if not e.is_directed:
                self.adj_list[e.to_vertex].append(e.from_vertex)

    def add_vertex(self, v):
        """
        Adds a vertex to the graph.

        Parameters:
        - v (Vertex): The vertex to add.

        Returns:
        - None

        Detailed Explanation:
        This method adds a new vertex to the adjacency list. If the vertex already exists, it does nothing.

        Implementation Steps:
        1. Check if the vertex 'v' is not already in the adjacency list.
        2. If not, add 'v' to the adjacency list with an empty list of neighbors.

        Example:
            v3 = Vertex(3, 'C')
            adj_list.add_vertex(v3)
            # Now adj_list.adj_list includes v3: []
        """
        if v not in self.adj_list:
            self.adj_list[v] = []
        else:
            raise ValueError('Already in graph')

    def remove_vertex(self, v):
        """
        Removes a vertex and all edges associated with it from the graph.

        Parameters:
        - v (Vertex): The vertex to remove.

        Returns:
        - None

        Detailed Explanation:
        This method removes the specified vertex from the adjacency list and also removes it from the neighbor lists of other vertices.

        Implementation Steps:
        1. Remove the vertex 'v' from the adjacency list.
        2. Iterate over all the neighbor lists in the adjacency list:
           - Remove 'v' from any neighbor lists where it appears.

        Example:
            adj_list.remove_vertex(v1)
            # Vertex v1 and all edges connected to it are removed from adj_list.adj_list.
        """
        if v in self.adj_list:
            del self.adj_list[v]
            for neighbors in self.adj_list.values():
                while v in neighbors:
                    neighbors.remove(v)

    def add_edge(self, e):
        """
        Adds an edge to the graph.

        Parameters:
        - e (Edge): The edge to add.

        Returns:
        - None

        Detailed Explanation:
        This method updates the adjacency list to include the new edge, considering whether it's directed or undirected.

        Implementation Steps:
        1. Add 'e.to_vertex' to the adjacency list of 'e.from_vertex'.
        2. If the edge is undirected, also add 'e.from_vertex' to the adjacency list of 'e.to_vertex'.

        Example:
            e2 = Edge(v1, v3)
            adj_list.add_edge(e2)
            # adj_list.adj_list[v1] will now include v3.
        """
        self.adj_list[e.from_vertex].append(e.to_vertex)
        if not e.is_directed:
            self.adj_list[e.to_vertex].append(e.from_vertex)

    def remove_edge(self, e):
        """
        Removes an edge from the graph.

        Parameters:
        - e (Edge): The edge to remove.

        Returns:
        - None

        Detailed Explanation:
        This method updates the adjacency list to remove the specified edge, considering whether it's directed or undirected.

        Implementation Steps:
        1. Remove 'e.to_vertex' from the adjacency list of 'e.from_vertex' if it exists.
        2. If the edge is undirected, also remove 'e.from_vertex' from the adjacency list of 'e.to_vertex'.

        Example:
            adj_list.remove_edge(e1)
            # The edge from v1 to v2 is removed from adj_list.adj_list.
        """
        if e.to_vertex in self.adj_list[e.from_vertex]:
            self.adj_list[e.from_vertex].remove(e.to_vertex)
        if not e.is_directed and e.from_vertex in self.adj_list[e.to_vertex]:
            self.adj_list[e.to_vertex].remove(e.from_vertex)

    def get_vertices(self):
        """
        Retrieves all vertices in the graph.

        Parameters:
        - None

        Returns:
        - list of Vertex: A list of all vertices in the graph.

        Implementation Steps:
        1. Return the list of keys from the adjacency list dictionary.

        Example:
            vertices = adj_list.get_vertices()
            # vertices will be a list of all Vertex instances in the graph.
        """
        return list(self.adj_list.keys())

    def get_edges(self):
        """
        Retrieves all edges in the graph.

        Parameters:
        - None

        Returns:
        - list of Edge: A list of all edges in the graph.

        Detailed Explanation:
        This method reconstructs the list of edges by examining the adjacency list.

        Implementation Steps:
        1. Initialize an empty list to store edges.
        2. Iterate over each vertex 'v' and its neighbors in the adjacency list:
           - For each neighbor, create an Edge instance from 'v' to the neighbor.
           - Append the Edge instance to the edges list.
        3. For undirected graphs, ensure that each edge is only added once to avoid duplicates.

        Example:
            edges = adj_list.get_edges()
            # edges will contain all Edge instances in the graph.
        """
        edges = []
        seen = set()
        for v, neighbors in self.adj_list.items():
            for neighbor in neighbors:
                edge = Edge(v, neighbor)
                if (neighbor, v) not in seen:
                    edges.append(edge)
                    seen.add((v, neighbor))
        return edges

    def get_neighbors(self, v):
        """
        Retrieves the neighbors of a given vertex.

        Parameters:
        - v (Vertex): The vertex whose neighbors are to be retrieved.

        Returns:
        - list of Vertex: A list of neighboring vertices.

        Implementation Steps:
        1. Return the adjacency list for the vertex 'v'.

        Example:
            neighbors = adj_list.get_neighbors(v1)
            # neighbors will be a list of Vertex instances adjacent to v1.
        """
        return self.adj_list.get(v, [])

class AdjMatrix:
    """
    Represents an adjacency matrix for graph representation.

    Attributes:
    - vertices (list): List of vertices.
    - matrix (list of lists): A 2D list representing the adjacency matrix.

    Detailed Explanation:
    An adjacency matrix uses a 2D array to represent a graph, where each cell [i][j] indicates the presence (and possibly weight) of an edge from vertex i to vertex j. It's efficient for dense graphs.

    Practical Usages:
    Adjacency matrices are suitable when the graph is dense, and quick edge existence checks are required, such as in network routing algorithms.

    """
    def __init__(self, V, E):
        """
        Initializes the adjacency matrix representation of a graph.

        Parameters:
        - V (list of Vertex): A list of Vertex instances representing the vertices of the graph.
        - E (list of Edge): A list of Edge instances representing the edges of the graph.

        Returns:
        - None

        Detailed Explanation:
        This method creates a 2D matrix to represent the graph. Each cell [i][j] corresponds to an edge from vertex 'i' to vertex 'j'.

        Implementation Steps:
        1. Store a copy of the vertices list.
        2. Create a mapping from each vertex to its index in the list.
        3. Initialize an n x n matrix with zeros, where n is the number of vertices.
        4. Iterate over each edge in E:
           - Set matrix[i][j] = 1, where 'i' and 'j' are indices of 'from_vertex' and 'to_vertex'.
           - If the edge is undirected, also set matrix[j][i] = 1.

        Example:
            v1 = Vertex(1, 'A')
            v2 = Vertex(2, 'B')
            e1 = Edge(v1, v2)
            adj_matrix = AdjMatrix([v1, v2], [e1])
            # adj_matrix.matrix will be [[0, 1], [0, 0]] for a directed edge from 'A' to 'B'.
        """
        self.vertices = V.copy()
        self.vertex_indices = {v: i for i, v in enumerate(V)}
        n = len(V)
        self.matrix = [[0] * n for _ in range(n)]
        for e in E:
            i = self.vertex_indices[e.from_vertex]
            j = self.vertex_indices[e.to_vertex]
            self.matrix[i][j] = 1
            if not e.is_directed:
                self.matrix[j][i] = 1

    def add_vertex(self, v):
        """
        Adds a vertex to the graph.

        Parameters:
        - v (Vertex): The vertex to add.

        Returns:
        - None

        Detailed Explanation:
        This method adds a new vertex to the vertices list and updates the adjacency matrix accordingly.

        Implementation Steps:
        1. Check if the vertex 'v' is not already in the vertex indices.
        2. Append 'v' to the vertices list.
        3. Update the vertex indices mapping.
        4. Add a new row and column to the matrix:
           - Append '0' to each existing row.
           - Append a new row with zeros.

        Example:
            v3 = Vertex(3, 'C')
            adj_matrix.add_vertex(v3)
            # The matrix now includes 'v3' as a new row and column.
        """
        if v not in self.vertex_indices:
            self.vertex_indices[v] = len(self.vertices)
            self.vertices.append(v)
            n = len(self.vertices)
            for row in self.matrix:
                row.append(0)
            self.matrix.append([0] * n)

    def remove_vertex(self, v):
        """
        Removes a vertex and all edges associated with it from the graph.

        Parameters:
        - v (Vertex): The vertex to remove.

        Returns:
        - None

        Detailed Explanation:
        This method removes the vertex from the vertices list, updates the matrix, and rebuilds the vertex indices mapping.

        Implementation Steps:
        1. Find the index of 'v' in the vertices list.
        2. Remove 'v' from the vertices list.
        3. Remove the corresponding row and column from the matrix.
        4. Rebuild the vertex indices mapping.

        Example:
            adj_matrix.remove_vertex(v1)
            # Vertex 'v1' and its edges are removed from the graph.
        """
        if v in self.vertex_indices:
            idx = self.vertex_indices[v]
            self.vertices.pop(idx)
            self.matrix.pop(idx)
            for row in self.matrix:
                row.pop(idx)
            self.vertex_indices = {vertex: i for i, vertex in enumerate(self.vertices)}

    def add_edge(self, e):
        """
        Adds an edge to the graph.

        Parameters:
        - e (Edge): The edge to add.

        Returns:
        - None

        Detailed Explanation:
        This method updates the matrix to represent the new edge.

        Implementation Steps:
        1. Retrieve the indices of 'from_vertex' and 'to_vertex'.
        2. Set matrix[i][j] = 1.
        3. If the edge is undirected, also set matrix[j][i] = 1.

        Example:
            e2 = Edge(v2, v3)
            adj_matrix.add_edge(e2)
            # The matrix is updated to include the new edge.
        """
        i = self.vertex_indices[e.from_vertex]
        j = self.vertex_indices[e.to_vertex]
        self.matrix[i][j] = 1
        if not e.is_directed:
            self.matrix[j][i] = 1

    def remove_edge(self, e):
        """
        Removes an edge from the graph.

        Parameters:
        - e (Edge): The edge to remove.

        Returns:
        - None

        Detailed Explanation:
        This method updates the matrix to remove the specified edge.

        Implementation Steps:
        1. Retrieve the indices of 'from_vertex' and 'to_vertex'.
        2. Set matrix[i][j] = 0.
        3. If the edge is undirected, also set matrix[j][i] = 0.

        Example:
            adj_matrix.remove_edge(e1)
            # The edge is removed from the matrix.
        """
        i = self.vertex_indices[e.from_vertex]
        j = self.vertex_indices[e.to_vertex]
        self.matrix[i][j] = 0
        if not e.is_directed:
            self.matrix[j][i] = 0

    def get_vertices(self):
        """
        Retrieves all vertices in the graph.

        Parameters:
        - None

        Returns:
        - list of Vertex: A list of all vertices in the graph.

        Implementation Steps:
        1. Return the vertices list.

        Example:
            vertices = adj_matrix.get_vertices()
            # vertices will be a list of all Vertex instances in the graph.
        """
        return self.vertices

    def get_edges(self):
        """
        Retrieves all edges in the graph.

        Parameters:
        - None

        Returns:
        - list of Edge: A list of all edges in the graph.

        Detailed Explanation:
        This method reconstructs the list of edges by examining the adjacency matrix.

        Implementation Steps:
        1. Initialize an empty list to store edges.
        2. Iterate over the matrix:
           - For each cell matrix[i][j] == 1:
             - Create an Edge instance from vertices[i] to vertices[j].
             - Append the Edge instance to the edges list.
        3. For undirected graphs, ensure that each edge is only added once to avoid duplicates.

        Example:
            edges = adj_matrix.get_edges()
            # edges will contain all Edge instances in the graph.
        """
        edges = []
        n = len(self.vertices)
        for i in range(n):
            for j in range(n):
                if self.matrix[i][j]:
                    edges.append(Edge(self.vertices[i], self.vertices[j]))
        return edges

    def get_neighbors(self, v):
        """
        Retrieves the neighbors of a given vertex.

        Parameters:
        - v (Vertex): The vertex whose neighbors are to be retrieved.

        Returns:
        - list of Vertex: A list of neighboring vertices.

        Implementation Steps:
        1. Retrieve the index 'idx' of the vertex 'v'.
        2. Iterate over the row matrix[idx]:
           - For each cell where matrix[idx][j] == 1:
             - Append vertices[j] to the neighbors list.

        Example:
            neighbors = adj_matrix.get_neighbors(v1)
            # neighbors will be a list of Vertex instances adjacent to v1.
        """
        idx = self.vertex_indices[v]
        neighbors = []
        n = len(self.vertices)
        for j in range(n):
            if self.matrix[idx][j]:
                neighbors.append(self.vertices[j])
        return neighbors
