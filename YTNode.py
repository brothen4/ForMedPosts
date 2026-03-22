import networkx as nx
import matplotlib.pyplot as plt
import random

# Create graph
G = nx.Graph()

# Top 50 YouTube channels (simplified names)
channels = [
"MrBeast","T-Series","Cocomelon","SET India","Vlad and Niki","Kids Diana Show",
"Stokes Twins","KIMPRO","Like Nastya","Zee Music Company",
"WWE","PewDiePie","Alejo Igoa","Goldmines","Sony SAB",
"BLACKPINK","Alan's Universe","ChuChu TV","Zee TV","A4",
"Topper Guild","Baby Shark - Pinkfong","BANGTANTV","Justin Bieber",
"Canal KondZilla","HYBE LABELS","Shemaroo Filmi Gaane","EminemMusic",
"Badabun","Dude Perfect","Felipe Neto","Mark Rober",
"Wave Music","Sony Music India","Speed Records","Movieclips",
"Toys and Colors","Marshmello","Taylor Swift","Ed Sheeran",
"CarryMinati","El Reino Infantil","Katy Perry","Ozuna",
"Whindersson Nunes","Rihanna","Billie Eilish","JuegaGerman",
"Mikecrack","Luisito Comunica"
]

G.add_nodes_from(channels)

# Randomly generate connections to simulate collaborations / audience overlap
for channel in channels:
    connections = random.sample(channels, random.randint(2,5))
    for c in connections:
        if channel != c:
            G.add_edge(channel, c)

# Network metrics
degree = nx.degree_centrality(G)
betweenness = nx.betweenness_centrality(G)
pagerank = nx.pagerank(G)

# Sort nodes by importance
sorted_channels = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)

print("\nTop 10 Most Important Channels (PageRank)\n")
print(f"{'Channel':<25}{'PageRank'}")
print("-"*40)

for channel, score in sorted_channels[:10]:
    print(f"{channel:<25}{score:.3f}")

# Most important node
most_important = sorted_channels[0][0]
print("\nMost Important Channel:", most_important)

# -------- Visualization -------- #

pos = nx.spring_layout(G, seed=42)

node_sizes = [pagerank[node]*8000 for node in G.nodes()]
node_colors = ["red" if node == most_important else "skyblue" for node in G.nodes()]

plt.figure(figsize=(12,9))

nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors)
nx.draw_networkx_edges(G, pos, alpha=0.3)
nx.draw_networkx_labels(G, pos, font_size=8)

plt.title("Network of Top 50 YouTube Channels\n(Node size = PageRank importance)")
plt.axis("off")

plt.show()