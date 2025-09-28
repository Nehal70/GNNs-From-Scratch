import heapq

"""
All Implementations are in line with CS3510 - Design and Analysis of Algorithms course at Georgia Institute of Technology.
"""

#Graph Traversals

#adjacency matrix representation
#rows are numbered from 0 to n-1 and columns are also numbered 0 to n-1
#graph[3,2] represents the edge from node 3 to node 2. If it is existent, it is equal to 1, else 0.
graph = [
    [0, 1, 1, 0, 0],  # Node 0 connects to 1, 2
    [1, 0, 1, 1, 0],  # Node 1 connects to 0, 2, 3  
    [1, 1, 0, 1, 1],  # Node 2 connects to 0, 1, 3, 4
    [0, 1, 1, 0, 1],  # Node 3 connects to 1, 2, 4
    [0, 0, 1, 1, 0]   # Node 4 connects to 2, 3
]


# Depth First Search
def depth_first_search(graph):
    """
    Parent method to perform a depth-first-search for a graph.
    """
    #initialise all nodes as unvisited
    visited = [False] * len(graph)
    #set clock for pre/post visit as 0
    clock = [0]
    #array where previsit[i] is previsit value of node i
    previsited = [0] * len(graph)
    #array where postvisit[i] is postvisit value of node i
    postvisited = [0] * len(graph)

    def previsit(node):
        previsited[node] = clock[0]
        clock[0] += 1

    def postvisit(node):
        postvisited[node] = clock[0]
        clock[0] += 1

    def explore(node):
        """
        Explore a node and its neighbors recursively.
        """
        visited[node] = True
        previsit(node)
        for j in range(len(graph[node])):
            if graph[node][j] == 1:
                #edge exists
                if not visited[j]:
                    explore(j)
            else:
                #edge doesn't exist
                continue
        postvisit(node)

    #for graphs where unable to reach all vertices.
    for i in range(len(graph)):
        if not visited[i]:
            explore(i)

#breadth first search
def breadth_first_search(graph):
    """
    Performing bfs.
    """
    n = len(graph)
    visited = [False] * n

    for i in range(len(graph)):
        #for each node as starting point
        if not visited[i]:
            dist = [float('inf')] * len(graph)
            dist[i] = 0

            queue = [i]

            while queue:
                u = queue[0]
                visited[u] = True
                queue = queue[1:]
                for j in range(len(graph[u])):
                    if graph[u][j] == 1 and dist[j] == float('inf'):
                        queue.append(j)
                        dist[j] = dist[u] + 1

#Djikstra's Algorithm
def Dijkstras(graph, start):
    """
    shortest path b/w 2 nodes in a graph (assuming no negative weights).
    """
    pq = []
    visited = set()
    dist = [float('inf')] * len(graph)
    prev = [None] * len(graph)
    dist[start] = 0
    pq.heappush(pq, (0, start))
    while pq:
        distance, u = heapq.heappop(pq) #shortest distance node gets removed
        visited[u] = True
        for j in range(len(graph[u])):
            #graph[u][j] is neighbour
            if graph[u][j] == 0:
                alt = dist[u] + graph[u][j]
                if alt < dist[j]:
                    dist[j] = alt
                    prev[j] = u
                heapq.heappush(pq, (alt, j))
    return dist, prev

#Boruvka's Algorithm
def boruvkas(graph):
    """
    MST via Boruvka's Algo.
    """
    MST = {}

    components = {}
    for i in range(len(graph)): #iterating over vertices
        components[i] = i #populating components with vertices

    while set(components.values()) > 1:
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
            if edge_key not in MST:
                MST[edge_key] = weight
                
                # Merge components: update all vertices in component2 to point to component1
                old_component = components[vertex2]
                new_component = components[vertex1]
                
                for v in range(len(components)):
                    if components[v] == old_component:
                        components[v] = new_component
    
    return MST

    
def kruskals(edges, num_vertices):
    """
    mst via kruskal's
    """
    mst = []
    parent = list(range(num_vertices))  # each vertex is its own parent initially
    
    def find(x):
        # find the root parent of vertex x
        if parent[x] != x:
            parent[x] = find(parent[x])  # path compression
        return parent[x]
    
    def union(x, y):
        # connect two components
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False  # already connected, would make cycle
    
    # sort all edges by weight, cheapest first
    edges.sort(key=lambda x: x[2])
    
    for u, v, weight in edges:
        if union(u, v):  # if connecting doesn't make cycle
            mst.append((u, v, weight))
            if len(mst) == num_vertices - 1:  # tree complete
                break
    
    return mst


#Prim's Algorithm
def prims(graph):
    """
    mst via prim's - grow tree from one vertex, always pick cheapest edge to expand
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
        
        # add all neighbors of current vertex to heap
        for neighbor in range(len(graph)):
            if neighbor not in visited and graph[vertex][neighbor] > 0:
                heapq.heappush(heap, (graph[vertex][neighbor], neighbor, vertex))
    
    return mst
