#include <vector>
#include <queue>
#include <set>
#include <map>
#include <algorithm>
#include <climits>
#include <iostream>
#include <functional>

/*
All Implementations are in line with CS3510 - Design and Analysis of Algorithms course at Georgia Institute of Technology.
*/

//Graph Traversals

//adjacency matrix representation
//rows are numbered from 0 to n-1 and columns are also numbered 0 to n-1
//graph[3][2] represents the edge from node 3 to node 2. If it is existent, it is equal to 1, else 0.
std::vector<std::vector<int>> graph = {
    {0, 1, 1, 0, 0},  // Node 0 connects to 1, 2
    {1, 0, 1, 1, 0},  // Node 1 connects to 0, 2, 3  
    {1, 1, 0, 1, 1},  // Node 2 connects to 0, 1, 3, 4
    {0, 1, 1, 0, 1},  // Node 3 connects to 1, 2, 4
    {0, 0, 1, 1, 0}   // Node 4 connects to 2, 3
};

// Depth First Search
void depth_first_search(const std::vector<std::vector<int>>& graph) {
    /*
    Parent method to perform a depth-first-search for a graph.
    */
    //initialise all nodes as unvisited
    std::vector<bool> visited(graph.size(), false);
    //set clock for pre/post visit as 0
    int clock = 0;
    //array where previsit[i] is previsit value of node i
    std::vector<int> previsited(graph.size(), 0);
    //array where postvisit[i] is postvisit value of node i
    std::vector<int> postvisited(graph.size(), 0);

    auto previsit = [&](int node) {
        previsited[node] = clock;
        clock++;
    };

    auto postvisit = [&](int node) {
        postvisited[node] = clock;
        clock++;
    };

    std::function<void(int)> explore = [&](int node) {
        /*
        Explore a node and its neighbors recursively.
        */
        visited[node] = true;
        previsit(node);
        for (int j = 0; j < graph[node].size(); j++) {
            if (graph[node][j] == 1) {
                //edge exists
                if (!visited[j]) {
                    explore(j);
                }
            } else {
                //edge doesn't exist
                continue;
            }
        }
        postvisit(node);
    };

    //for graphs where unable to reach all vertices.
    for (int i = 0; i < graph.size(); i++) {
        if (!visited[i]) {
            explore(i);
        }
    }
}

//breadth first search
void breadth_first_search(const std::vector<std::vector<int>>& graph) {
    /*
    Performing bfs.
    */
    int n = graph.size();
    std::vector<bool> visited(n, false);

    for (int i = 0; i < graph.size(); i++) {
        //for each node as starting point
        if (!visited[i]) {
            std::vector<int> dist(graph.size(), INT_MAX);
            dist[i] = 0;

            std::queue<int> queue;
            queue.push(i);

            while (!queue.empty()) {
                int u = queue.front();
                visited[u] = true;
                queue.pop();
                for (int j = 0; j < graph[u].size(); j++) {
                    if (graph[u][j] == 1 && dist[j] == INT_MAX) {
                        queue.push(j);
                        dist[j] = dist[u] + 1;
                    }
                }
            }
        }
    }
}

//Djikstra's Algorithm
std::pair<std::vector<int>, std::vector<int>> dijkstras(const std::vector<std::vector<int>>& graph, int start) {
    /*
    shortest path b/w 2 nodes in a graph (assuming no negative weights).
    */
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<std::pair<int, int>>> pq;
    std::set<int> visited;
    std::vector<int> dist(graph.size(), INT_MAX);
    std::vector<int> prev(graph.size(), -1);
    dist[start] = 0;
    pq.push({0, start});
    
    while (!pq.empty()) {
        auto [distance, u] = pq.top(); //shortest distance node gets removed
        pq.pop();
        visited.insert(u);
        
        for (int j = 0; j < graph[u].size(); j++) {
            //graph[u][j] is neighbour
            if (graph[u][j] != 0) {
                int alt = dist[u] + graph[u][j];
                if (alt < dist[j]) {
                    dist[j] = alt;
                    prev[j] = u;
                    pq.push({alt, j});
                }
            }
        }
    }
    return {dist, prev};
}

//Boruvka's Algorithm
std::map<std::pair<int, int>, int> boruvkas(const std::vector<std::vector<int>>& graph) {
    /*
    MST via Boruvka's Algo.
    */
    std::map<std::pair<int, int>, int> MST;

    std::vector<int> components(graph.size());
    for (int i = 0; i < graph.size(); i++) { //iterating over vertices
        components[i] = i; //populating components with vertices
    }

    while (std::set<int>(components.begin(), components.end()).size() > 1) {
        std::map<int, std::tuple<int, int, int>> lightest_edges;
        
        for (int vertex = 0; vertex < graph.size(); vertex++) {
            for (int edge = 0; edge < graph[vertex].size(); edge++) {
                int neighbour = edge;
                int weight = graph[vertex][edge];

                if (weight == 0 || weight == INT_MAX) {
                    continue;
                }

                int component1 = components[vertex];
                int component2 = components[neighbour];

                if (component1 != component2) {
                    if (lightest_edges.find(component1) == lightest_edges.end() || 
                        weight < std::get<2>(lightest_edges[component1])) {
                        lightest_edges[component1] = std::make_tuple(vertex, neighbour, weight);
                    }
                }
            }
        }

        for (auto& [component, edge] : lightest_edges) {
            auto [vertex1, vertex2, weight] = edge;
            
            // Create edge key (order doesn't matter for undirected graph)
            std::pair<int, int> edge_key = {std::min(vertex1, vertex2), std::max(vertex1, vertex2)};
            
            // Add to MST if not already there
            if (MST.find(edge_key) == MST.end()) {
                MST[edge_key] = weight;
                
                // Merge components: update all vertices in component2 to point to component1
                int old_component = components[vertex2];
                int new_component = components[vertex1];
                
                for (int v = 0; v < components.size(); v++) {
                    if (components[v] == old_component) {
                        components[v] = new_component;
                    }
                }
            }
        }
    }
    
    return MST;
}

std::vector<std::tuple<int, int, int>> kruskals(std::vector<std::tuple<int, int, int>> edges, int num_vertices) {
    /*
    mst via kruskal's
    */
    std::vector<std::tuple<int, int, int>> mst;
    std::vector<int> parent(num_vertices);
    for (int i = 0; i < num_vertices; i++) {
        parent[i] = i;  // each vertex is its own parent initially
    }
    
    std::function<int(int)> find = [&](int x) -> int {
        // find the root parent of vertex x
        if (parent[x] != x) {
            parent[x] = find(parent[x]);  // path compression
        }
        return parent[x];
    };
    
    auto union_sets = [&](int x, int y) -> bool {
        // connect two components
        int px = find(x), py = find(y);
        if (px != py) {
            parent[px] = py;
            return true;
        }
        return false;  
    };
    
    // sort all edges by weight, cheapest first
    std::sort(edges.begin(), edges.end(), [](const auto& a, const auto& b) {
        return std::get<2>(a) < std::get<2>(b);
    });
    
    for (auto& [u, v, weight] : edges) {
        if (union_sets(u, v)) {  // if connecting doesn't make cycle
            mst.push_back({u, v, weight});
            if (mst.size() == num_vertices - 1) {  // tree complete
                break;
            }
        }
    }
    
    return mst;
}

//Prim's Algorithm
std::vector<std::tuple<int, int, int>> prims(const std::vector<std::vector<int>>& graph) {
    /*
    mst via prim's 
    */
    std::vector<std::tuple<int, int, int>> mst;
    std::set<int> visited;
    std::priority_queue<std::tuple<int, int, int>, std::vector<std::tuple<int, int, int>>, 
                       std::greater<std::tuple<int, int, int>>> heap;
    heap.push({0, 0, -1});  // (weight, vertex, parent)
    
    while (!heap.empty()) {
        auto [weight, vertex, parent] = heap.top();
        heap.pop();
        
        if (visited.find(vertex) != visited.end()) {
            continue;  // already added this vertex
        }
            
        visited.insert(vertex);
        if (parent != -1) {  // not the starting vertex
            mst.push_back({parent, vertex, weight});
        }
        
        // add all neighbors of current vertex to heap
        for (int neighbor = 0; neighbor < graph.size(); neighbor++) {
            if (visited.find(neighbor) == visited.end() && graph[vertex][neighbor] > 0) {
                heap.push({graph[vertex][neighbor], neighbor, vertex});
            }
        }
    }
    
    return mst;
}
