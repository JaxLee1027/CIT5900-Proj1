import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer

file_path = "fsrdc_outputs.csv"
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: The file {file_path} was not found.")
    exit(1)
except pd.errors.EmptyDataError:
    print(f"Error: The file {file_path} is empty.")
    exit(1)
except pd.errors.ParserError:
    print(f"Error: The file {file_path} has formatting issues.")
    exit(1)

# Extract required columns and rename them according to project requirements
df_filtered = df[['Title', 'Year', 'Agency', 'Keywords']].copy()

# Graph construction using networkx
G = nx.Graph()

# Add nodes (each research output as a node)
for index, row in df_filtered.iterrows():
    G.add_node(row['Title'], keywords=row['Keywords'])

# Function to check if two research outputs share at least 1 common keywords
def share_common_keywords(output1, output2):
    if not output1['keywords'] or not output2['keywords']:
        return False
    keywords1 = set(output1['keywords'].split('; '))
    keywords2 = set(output2['keywords'].split('; '))
    return len(keywords1.intersection(keywords2)) >= 1  # Requires at least 1 shared keyword

# Add edges between nodes if they share common attributes
titles = list(df_filtered['Title'])
for i in range(len(titles)):
    for j in range(i + 1, len(titles)):
        if share_common_keywords(G.nodes[titles[i]], G.nodes[titles[j]]):
            G.add_edge(titles[i], titles[j])

assert G.number_of_nodes() > 0, "Graph should have at least one node"
assert G.number_of_edges() >= 0, "Edges should be non-negative"

# Display graph statistics
graph_info = {
    "Total Nodes": G.number_of_nodes(),
    "Total Edges": G.number_of_edges(),
    "Is Directed": nx.is_directed(G),
    "Is Connected": nx.is_connected(G) if G.number_of_nodes() > 0 else False,
}

# Save the graph for visualization if needed
nx.write_gexf(G, "fsrdc_outputs_graph.gexf")

# Display graph summary
print(graph_info)

# Implement BFS (Breadth-First Search)
from collections import deque

def bfs(graph, start_node):
    visited = set()
    queue = deque([start_node])
    visited.add(start_node)

    traversal_order = []

    while queue:
        current = queue.popleft()
        traversal_order.append(current)

        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return traversal_order

import random
# Implement DFS (Depth-First Search)
def dfs_randomized(graph, start_node, visited=None):
    if visited is None:
        visited = set()
    visited.add(start_node)
    traversal_order = [start_node]

    neighbors = list(graph.neighbors(start_node))
    random.shuffle(neighbors)  # Shuffle the order of neighbor visits

    for neighbor in neighbors:
        if neighbor not in visited:
            traversal_order.extend(dfs_randomized(graph, neighbor, visited))
    return traversal_order

# Select a random starting node for traversal
largest_component = max(nx.connected_components(G), key=len)
start_node = list(largest_component)[0]
print(f"Total Nodes: {G.number_of_nodes()}, Total Edges: {G.number_of_edges()}")
print('largest_component:',len(largest_component))
print('Z:',nx.number_connected_components(G))
# Perform BFS and DFS if a starting node exists
if start_node:
    bfs_result = bfs(G, start_node)
    dfs_result = dfs_randomized(G, start_node)
else:
    bfs_result = []
    dfs_result = []
    
assert len(bfs_result) == len(dfs_result), "BFS and DFS should visit the same number of nodes"

# Save traversal results for review
bfs_output_file = "bfs_traversal.txt"
dfs_output_file = "dfs_traversal.txt"

with open(bfs_output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(bfs_result))

with open(dfs_output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(dfs_result))

# Display BFS and DFS traversal results summary
traversal_summary = {
    "BFS Traversal (First 10 Nodes)": bfs_result[:10],
    "DFS Traversal (First 10 Nodes)": dfs_result[:10],
    "Total Nodes Traversed in BFS": len(bfs_result),
    "Total Nodes Traversed in DFS": len(dfs_result)
}

print(traversal_summary)