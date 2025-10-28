import heapq

"""
Graph Algorithms Class
All Implementations are in line with CS3510 - Design and Analysis of Algorithms course at Georgia Institute of Technology.

Usage:
    from gnns_ml import GraphAlgorithms
    
    ga = GraphAlgorithms()
    result = ga.depth_first_search(graph)
    result = ga.dijkstra(graph, start)
"""


class GraphAlgorithms:
    """
    A collection of classic graph algorithms implemented from scratch.
    
    Includes:
    - Traversal: DFS, BFS
    - Shortest Path: Dijkstra's
    - Minimum Spanning Tree: Kruskal's, Prim's, Boruvka's
    """
    
    def __init__(self):
        """Initialize the GraphAlgorithms class."""
        pass
    
    def depth_first_search(self, graph):
        """
        Perform a depth-first search (DFS) on a graph.
        
        Args:
            graph: Adjacency matrix representation of the graph
            
        Returns:
            tuple: (previsited, postvisited) arrays tracking visit times
        """
        # Initialize all nodes as unvisited
        visited = [False] * len(graph)
        clock = [0]
        previsited = [0] * len(graph)
        postvisited = [0] * len(graph)

        def previsit(node):
            previsited[node] = clock[0]
            clock[0] += 1

        def postvisit(node):
            postvisited[node] = clock[0]
            clock[0] += 1

        def explore(node):
            """Explore a node and its neighbors recursively."""
            visited[node] = True
            previsit(node)
            for j in range(len(graph[node])):
                if graph[node][j] == 1:
                    # edge exists
                    if not visited[j]:
                        explore(j)
            postvisit(node)

        # For disconnected graphs, ensure we visit all vertices
        for i in range(len(graph)):
            if not visited[i]:
                explore(i)
        
        return previsited, postvisited

    def breadth_first_search(self, graph, start=None):
        """
        Perform breadth-first search (BFS) on a graph.
        
        Args:
            graph: Adjacency matrix representation of the graph
            start: Starting node (optional, searches from all nodes if not specified)
            
        Returns:
            dict: Dictionary mapping node to distance from start (or distances from all starts)
        """
        n = len(graph)
        visited = [False] * n
        all_distances = {}

        if start is not None:
            # Single-source BFS
            dist = [float('inf')] * len(graph)
            dist[start] = 0
            queue = [start]
            visited[start] = True

            while queue:
                u = queue[0]
                queue = queue[1:]
                for j in range(len(graph[u])):
                    if graph[u][j] == 1 and dist[j] == float('inf'):
                        queue.append(j)
                        dist[j] = dist[u] + 1
            
            return dist
        else:
            # Multi-source BFS
            for i in range(len(graph)):
                if not visited[i]:
                    dist = [float('inf')] * len(graph)
                    dist[i] = 0
                    queue = [i]
                    visited[i] = True

                    while queue:
                        u = queue[0]
                        visited[u] = True
                        queue = queue[1:]
                        for j in range(len(graph[u])):
                            if graph[u][j] == 1 and dist[j] == float('inf'):
                                queue.append(j)
                                dist[j] = dist[u] + 1
                    
                    all_distances[i] = dist
        
        return all_distances

    def dijkstra(self, graph, start):
        """
        Find shortest path between nodes using Dijkstra's algorithm.
        Assumes no negative weights.
        
        Args:
            graph: Weighted adjacency matrix representation
            start: Starting node
            
        Returns:
            tuple: (distances, predecessors) arrays
        """
        pq = []
        visited = set()
        dist = [float('inf')] * len(graph)
        prev = [None] * len(graph)
        dist[start] = 0
        heapq.heappush(pq, (0, start))
        
        while pq:
            distance, u = heapq.heappop(pq)  # shortest distance node gets removed
            if u in visited:
                continue
            visited.add(u)
            
            for j in range(len(graph[u])):
                if graph[u][j] > 0:  # edge exists
                    alt = dist[u] + graph[u][j]
                    if alt < dist[j]:
                        dist[j] = alt
                        prev[j] = u
                        heapq.heappush(pq, (alt, j))
        
        return dist, prev

    def boruvka(self, graph):
        """
        Find Minimum Spanning Tree (MST) using Boruvka's algorithm.
        
        Args:
            graph: Weighted adjacency matrix
            
        Returns:
            dict: MST edges with weights (edge tuples as keys)
        """
        mst = {}
        components = {}
        
        for i in range(len(graph)):  # iterating over vertices
            components[i] = i  # populating components with vertices

        while len(set(components.values())) > 1:
            lightest_edges = {}
            
            for vertex in range(len(graph)):
                for edge in range(len(graph[vertex])):
                    neighbour = edge
                    weight = graph[vertex][edge]

                    if weight == 0 or weight == float('inf'):
                        continue

                    component1 = components[vertex]
                    component2 = components[neighbour]

                    if component1 != component2:
                        if component1 not in lightest_edges or weight < lightest_edges[component1][2]:
                            lightest_edges[component1] = (vertex, neighbour, weight)

            for component, edge in lightest_edges.items():
                vertex1, vertex2, weight = edge
                
                # Create edge key (order doesn't matter for undirected graph)
                edge_key = tuple(sorted([vertex1, vertex2]))
                
                # Add to MST if not already there
                if edge_key not in mst:
                    mst[edge_key] = weight
                    
                    # Merge components: update all vertices in component2 to point to component1
                    old_component = components[vertex2]
                    new_component = components[vertex1]
                    
                    for v in range(len(components)):
                        if components[v] == old_component:
                            components[v] = new_component
        
        return mst

    def kruskal(self, edges, num_vertices):
        """
        Find Minimum Spanning Tree (MST) using Kruskal's algorithm.
        
        Args:
            edges: List of tuples (u, v, weight) representing edges
            num_vertices: Number of vertices in the graph
            
        Returns:
            list: MST edges as tuples (u, v, weight)
        """
        mst = []
        parent = list(range(num_vertices))  # each vertex is its own parent initially
        
        def find(x):
            """Find the root parent of vertex x."""
            if parent[x] != x:
                parent[x] = find(parent[x])  # path compression
            return parent[x]
        
        def union(x, y):
            """Connect two components."""
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False  # already connected, would make cycle
        
        # Sort all edges by weight, cheapest first
        edges.sort(key=lambda x: x[2])
        
        for u, v, weight in edges:
            if union(u, v):  # if connecting doesn't make cycle
                mst.append((u, v, weight))
                if len(mst) == num_vertices - 1:  # tree complete
                    break
        
        return mst

    def prim(self, graph):
        """
        Find Minimum Spanning Tree (MST) using Prim's algorithm.
        
        Args:
            graph: Weighted adjacency matrix
            
        Returns:
            list: MST edges as tuples (u, v, weight)
        """
        mst = []
        visited = set()
        heap = [(0, 0, -1)]  # (weight, vertex, parent)
        
        while heap:
            weight, vertex, parent = heapq.heappop(heap)
            
            if vertex in visited:
                continue  # already added this vertex
                
            visited.add(vertex)
            if parent != -1:  # not the starting vertex
                mst.append((parent, vertex, weight))
            
            # Add all neighbors of current vertex to heap
            for neighbor in range(len(graph)):
                if neighbor not in visited and graph[vertex][neighbor] > 0:
                    heapq.heappush(heap, (graph[vertex][neighbor], neighbor, vertex))
        
        return mst


# Backward compatibility: Keep old function names as module-level functions
# that create an instance and call the method
_ga = GraphAlgorithms()

def depth_first_search(graph):
    """Deprecated: Use GraphAlgorithms().depth_first_search() instead."""
    return _ga.depth_first_search(graph)

def breadth_first_search(graph):
    """Deprecated: Use GraphAlgorithms().breadth_first_search() instead."""
    return _ga.breadth_first_search(graph)

def Dijkstras(graph, start):
    """Deprecated: Use GraphAlgorithms().dijkstra() instead."""
    return _ga.dijkstra(graph, start)

def boruvkas(graph):
    """Deprecated: Use GraphAlgorithms().boruvka() instead."""
    return _ga.boruvka(graph)

def kruskals(edges, num_vertices):
    """Deprecated: Use GraphAlgorithms().kruskal() instead."""
    return _ga.kruskal(edges, num_vertices)

def prims(graph):
    """Deprecated: Use GraphAlgorithms().prim() instead."""
    return _ga.prim(graph)
