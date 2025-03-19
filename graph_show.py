import networkx as nx
import matplotlib.pyplot as plt

# Load the GEXF file
graph_path = "fsrdc_outputs_graph.gexf"
G = nx.read_gexf(graph_path)

# Plot the graph
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)  # Positioning algorithm
nx.draw(G, pos, with_labels=True, node_size=20, font_size=8)
plt.title("Research Outputs Graph")
plt.show()