"""
Network analysis utilities (centrality, communities, etc.).
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    logger.warning("NetworkX not installed. Network analysis unavailable.")
    NETWORKX_AVAILABLE = False


class NetworkAnalyzer:
    """
    Analyze network properties and structure.
    """
    
    def __init__(self):
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX is required for network analysis")
        
        logger.info("Initialized NetworkAnalyzer")
    
    def edgelist_to_graph(
        self,
        edgelist: pd.DataFrame,
        source_col: str,
        target_col: str,
        weight_col: Optional[str] = None
    ) -> nx.Graph:
        """
        Convert edgelist DataFrame to NetworkX graph.
        
        Args:
            edgelist: DataFrame with edges
            source_col: Source node column
            target_col: Target node column
            weight_col: Optional weight column
            
        Returns:
            NetworkX Graph object
        """
        G = nx.Graph()
        
        for _, row in edgelist.iterrows():
            source = row[source_col]
            target = row[target_col]
            
            if weight_col:
                weight = row[weight_col]
                G.add_edge(source, target, weight=weight)
            else:
                G.add_edge(source, target)
        
        logger.info(
            f"Created graph: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges"
        )
        
        return G
    
    def calculate_centrality(
        self,
        G: nx.Graph,
        method: str = "degree"
    ) -> pd.DataFrame:
        """
        Calculate node centrality measures.
        
        Args:
            G: NetworkX graph
            method: Centrality metric ('degree', 'betweenness', 'eigenvector', 'pagerank')
            
        Returns:
            DataFrame with nodes and centrality scores
        """
        logger.info(f"Calculating {method} centrality...")
        
        if method == "degree":
            centrality = nx.degree_centrality(G)
        elif method == "betweenness":
            centrality = nx.betweenness_centrality(G)
        elif method == "eigenvector":
            centrality = nx.eigenvector_centrality(G, max_iter=1000)
        elif method == "pagerank":
            centrality = nx.pagerank(G)
        else:
            raise ValueError(f"Unknown centrality method: {method}")
        
        df = pd.DataFrame([
            {'node': node, f'{method}_centrality': score}
            for node, score in centrality.items()
        ]).sort_values(f'{method}_centrality', ascending=False)
        
        logger.info(f"Top node: {df.iloc[0]['node']} (score={df.iloc[0][f'{method}_centrality']:.4f})")
        
        return df
    
    def detect_communities(
        self,
        G: nx.Graph,
        method: str = "louvain"
    ) -> Dict[str, int]:
        """
        Detect communities in the network.
        
        Args:
            G: NetworkX graph
            method: Community detection algorithm ('louvain', 'label_propagation')
            
        Returns:
            Dict mapping node to community ID
        """
        logger.info(f"Detecting communities with {method}...")
        
        if method == "louvain":
            try:
                import community as community_louvain
                communities = community_louvain.best_partition(G)
            except ImportError:
                logger.error("python-louvain not installed. Using label propagation instead.")
                method = "label_propagation"
        
        if method == "label_propagation":
            communities_gen = nx.community.label_propagation_communities(G)
            communities = {}
            for i, community_set in enumerate(communities_gen):
                for node in community_set:
                    communities[node] = i
        
        n_communities = len(set(communities.values()))
        logger.info(f"Found {n_communities} communities")
        
        return communities
    
    def get_network_stats(self, G: nx.Graph) -> Dict[str, float]:
        """
        Calculate basic network statistics.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Dict with network statistics
        """
        stats = {
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'density': nx.density(G),
            'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
        }
        
        if nx.is_connected(G):
            stats['avg_shortest_path'] = nx.average_shortest_path_length(G)
            stats['diameter'] = nx.diameter(G)
        else:
            stats['n_components'] = nx.number_connected_components(G)
            # Stats for largest component
            largest_cc = max(nx.connected_components(G), key=len)
            G_cc = G.subgraph(largest_cc)
            stats['largest_component_size'] = len(largest_cc)
            stats['avg_shortest_path_largest_cc'] = nx.average_shortest_path_length(G_cc)
        
        return stats
