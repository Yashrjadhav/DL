%%writefile BFSDFS.cpp

#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>
using namespace std;

class Graph {
    int V;
    vector<vector<int>> adj;

public:
    Graph(int V) : V(V), adj(V) {}
    void addEdge(int u, int v) { adj[u].push_back(v), adj[v].push_back(u); }

    void parallelBFS(int start) {
        vector<bool> vis(V, false); queue<int> q;
        vis[start] = true; q.push(start);
        cout << "Parallel BFS: ";
        while (!q.empty()) {
            int n = q.size(); vector<int> level(n);
            for (int i = 0; i < n; ++i) level[i] = q.front(), q.pop();
            #pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                int u = level[i];
                #pragma omp critical
                cout << u << " ";
                for (int v : adj[u]) {
                    if (!vis[v]) {
                        #pragma omp critical
                        if (!vis[v]) vis[v] = true, q.push(v);
                    }
                }
            }
        }
        cout << endl;
    }

    void parallelDFS(int start) {
        vector<bool> vis(V, false); stack<int> s; s.push(start);
        cout << "Parallel DFS: ";
        while (!s.empty()) {
            int u;
            #pragma omp critical
            { u = s.top(); s.pop(); }
            if (!vis[u]) {
                vis[u] = true;
                #pragma omp critical
                cout << u << " ";
                #pragma omp parallel for
                for (int i = 0; i < adj[u].size(); ++i) {
                    int v = adj[u][i];
                    if (!vis[v]) {
                        #pragma omp critical
                        s.push(v);
                    }
                }
            }
        }
        cout << endl;
    }
};

int main() {
    Graph g(6);
    g.addEdge(0,1); g.addEdge(0,2); g.addEdge(1,3);
    g.addEdge(1,4); g.addEdge(2,5);
    g.parallelBFS(0); g.parallelDFS(0);
}

//!g++ -fopenmp BFSDFS.cpp -o run
//!./run
