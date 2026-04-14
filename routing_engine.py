"""
Optimal Network Routing Engine
==============================
A realistic 15-city Indian ISP backbone with multi-attribute edges.

Link types:
  Fiber:     Moderate cost, very low latency, very high reliability
  Cable:     Low cost, moderate latency, moderate reliability
  Satellite: Cheapest, but extreme latency and poor reliability

The cheapest path often uses satellite links that violate real-world
QoS constraints — this is where the Genetic Algorithm shines by
solving the NP-Hard Constrained Shortest Path problem.
"""

import random
import time
import heapq
import copy
import math
import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA STRUCTURES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class EdgeAttr:
    cost: float
    latency: float       # milliseconds
    reliability: float   # 0.0 – 1.0
    link_type: str       # "Fiber", "Cable", "Satellite"


@dataclass
class PathMetrics:
    path: List[str]
    cost: float
    latency: float
    reliability: float
    valid: bool = True
    violation: str = ""


@dataclass
class GAConfig:
    pop_size: int = 60
    max_gen: int = 100
    cx_rate: float = 0.85
    mut_rate_init: float = 0.40
    mut_rate_min: float = 0.05
    tournament_k: int = 5
    elitism: int = 3
    diversity_threshold: float = 0.20
    restart_fraction: float = 0.25
    # Constraints
    max_latency: float = float('inf')
    min_reliability: float = 0.0
    # Multi-objective weights
    alpha: float = 1.0   # cost
    beta: float = 0.0    # latency
    gamma: float = 0.0   # (1-reliability)
    cost_only: bool = True


@dataclass
class GenSnapshot:
    gen: int
    best_cost: float
    avg_cost: float
    best_path: List[str]
    diversity: float
    wall_time: float


@dataclass
class GAResult:
    best_path: List[str]
    best_cost: float
    best_latency: float
    best_reliability: float
    time_taken: float
    generations: List[GenSnapshot]
    feasible: bool


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRAPH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Graph:
    """Directed graph with multi-attribute edges."""

    def __init__(self):
        self.adj: Dict[str, List[Tuple[str, EdgeAttr]]] = {}
        self.nodes: set = set()

    def add_edge(self, u, v, attr, bidirectional=True):
        self.adj.setdefault(u, []).append((v, attr))
        self.nodes.update([u, v])
        if bidirectional:
            self.adj.setdefault(v, []).append((u, attr))

    def neighbors(self, node):
        return self.adj.get(node, [])

    def edge_attr(self, u, v):
        for nbr, a in self.adj.get(u, []):
            if nbr == v:
                return a
        return None

    def path_cost(self, path):
        total = 0.0
        for i in range(len(path) - 1):
            a = self.edge_attr(path[i], path[i + 1])
            if a is None:
                return float('inf')
            total += a.cost
        return total

    def path_latency(self, path):
        total = 0.0
        for i in range(len(path) - 1):
            a = self.edge_attr(path[i], path[i + 1])
            if a is None:
                return float('inf')
            total += a.latency
        return total

    def path_reliability(self, path):
        rel = 1.0
        for i in range(len(path) - 1):
            a = self.edge_attr(path[i], path[i + 1])
            if a is None:
                return 0.0
            rel *= a.reliability
        return rel

    def is_valid_path(self, path, src, dst):
        if not path or path[0] != src or path[-1] != dst:
            return False
        if len(path) != len(set(path)):
            return False
        for i in range(len(path) - 1):
            if self.edge_attr(path[i], path[i + 1]) is None:
                return False
        return True

    def path_metrics(self, path, src, dst, max_lat=float('inf'), min_rel=0.0):
        if not self.is_valid_path(path, src, dst):
            return PathMetrics(path, float('inf'), float('inf'), 0.0, False, "Invalid path")
        c = self.path_cost(path)
        l = self.path_latency(path)
        r = self.path_reliability(path)
        violation = ""
        if l > max_lat:
            violation = f"Latency {l:.1f}ms exceeds limit {max_lat:.1f}ms"
        elif r < min_rel:
            violation = f"Reliability {r:.4f} below minimum {min_rel:.4f}"
        return PathMetrics(path, c, l, r, violation == "", violation)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIXED ISP BACKBONE NETWORK — INDIA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Geographic positions for plotly visualization (India map layout)
NODE_POSITIONS = {
    "DEL": (0.42, 0.88),    # Delhi
    "JAI": (0.30, 0.78),    # Jaipur
    "LKO": (0.58, 0.82),    # Lucknow
    "PAT": (0.72, 0.80),    # Patna
    "KOL": (0.82, 0.66),    # Kolkata
    "AMD": (0.18, 0.64),    # Ahmedabad
    "BHO": (0.40, 0.64),    # Bhopal
    "MUM": (0.14, 0.48),    # Mumbai
    "PUN": (0.22, 0.42),    # Pune
    "GOA": (0.20, 0.30),    # Goa
    "HYD": (0.44, 0.40),    # Hyderabad
    "VIZ": (0.65, 0.44),    # Visakhapatnam
    "BLR": (0.36, 0.20),    # Bangalore
    "CHN": (0.54, 0.22),    # Chennai
    "KOC": (0.30, 0.08),    # Kochi
    "IXC": (0.38, 0.94),    # Chandigarh
    "SXR": (0.32, 0.99),    # Srinagar
    "AGR": (0.45, 0.82),    # Agra
    "KNP": (0.52, 0.80),    # Kanpur
    "VNS": (0.62, 0.76),    # Varanasi
    "GAU": (0.92, 0.80),    # Guwahati
    "BBI": (0.76, 0.54),    # Bhubaneswar
    "IXR": (0.70, 0.64),    # Ranchi
    "IDR": (0.34, 0.60),    # Indore
    "STV": (0.16, 0.55),    # Surat
    "NAG": (0.46, 0.54),    # Nagpur
    "RPR": (0.58, 0.56),    # Raipur
    "TRV": (0.34, 0.02),    # Thiruvananthapuram
    "IXM": (0.44, 0.08),    # Madurai
    "CJB": (0.38, 0.12),    # Coimbatore
}

NODE_LABELS = {
    "DEL": "Delhi",
    "JAI": "Jaipur",
    "LKO": "Lucknow",
    "PAT": "Patna",
    "KOL": "Kolkata",
    "AMD": "Ahmedabad",
    "BHO": "Bhopal",
    "MUM": "Mumbai",
    "PUN": "Pune",
    "GOA": "Goa",
    "HYD": "Hyderabad",
    "VIZ": "Visakhapatnam",
    "BLR": "Bangalore",
    "CHN": "Chennai",
    "KOC": "Kochi",
    "IXC": "Chandigarh",
    "SXR": "Srinagar",
    "AGR": "Agra",
    "KNP": "Kanpur",
    "VNS": "Varanasi",
    "GAU": "Guwahati",
    "BBI": "Bhubaneswar",
    "IXR": "Ranchi",
    "IDR": "Indore",
    "STV": "Surat",
    "NAG": "Nagpur",
    "RPR": "Raipur",
    "TRV": "Thiruvananthapuram",
    "IXM": "Madurai",
    "CJB": "Coimbatore",
}

LINK_COLORS = {
    "Fiber": "#00C6FF",
    "Cable": "#F97316",
    "Satellite": "#EF4444",
}

# Pre-computed key routes for reference (DEL → CHN)
# Route 1 (Satellite):     cost=2,  lat=130ms, rel=0.840
# Route 2 (Central Cable): cost=11, lat=30ms,  rel=0.876
# Route 3 (Central+Fiber): cost=18, lat=30ms,  rel=0.910
# Route 4 (Eastern):       cost=23, lat=37ms,  rel=0.897
# Route 5 (Western Fiber): cost=30, lat=26ms,  rel=0.986


def build_network():
    """
    15-city Indian ISP backbone with deliberate multi-objective tradeoffs.

    Key insight: the cheapest DEL→CHN path uses a satellite shortcut
    (cost=2, latency=130ms, reliability=0.840). Any latency or
    reliability constraint will invalidate this path, creating an
    NP-Hard constrained routing problem where the GA excels.
    """
    g = Graph()

    edges = [
        # ─── Northern Network ───
        ("SXR", "IXC", EdgeAttr(7, 12, 0.985, "Cable")),
        ("IXC", "DEL", EdgeAttr(5, 5, 0.998, "Fiber")),
        ("DEL", "AGR", EdgeAttr(4, 3, 0.999, "Fiber")),
        ("AGR", "KNP", EdgeAttr(5, 4, 0.998, "Fiber")),
        ("KNP", "LKO", EdgeAttr(3, 3, 0.999, "Fiber")),
        ("LKO", "DEL", EdgeAttr(7, 10, 0.98, "Cable")),
        ("DEL", "JAI", EdgeAttr(6, 4, 0.995, "Fiber")),
        ("IXC", "JAI", EdgeAttr(8, 14, 0.97, "Cable")),
        ("SXR", "DEL", EdgeAttr(9, 11, 0.985, "Cable")), # New

        # ─── Eastern & North-Eastern Network ───
        ("LKO", "VNS", EdgeAttr(5, 4, 0.997, "Fiber")),
        ("VNS", "PAT", EdgeAttr(5, 4, 0.998, "Fiber")),
        ("PAT", "GAU", EdgeAttr(9, 8, 0.990, "Fiber")),
        ("PAT", "IXR", EdgeAttr(6, 5, 0.995, "Fiber")),
        ("KOL", "GAU", EdgeAttr(11, 15, 0.98, "Cable")),
        ("VNS", "IXR", EdgeAttr(6, 10, 0.975, "Cable")),
        ("IXR", "KOL", EdgeAttr(7, 6, 0.992, "Fiber")),
        ("KOL", "BBI", EdgeAttr(6, 5, 0.996, "Fiber")),
        ("IXR", "RPR", EdgeAttr(7, 11, 0.97, "Cable")),
        ("GAU", "BBI", EdgeAttr(10, 14, 0.975, "Cable")), # New
        ("PAT", "KOL", EdgeAttr(8, 7, 0.995, "Fiber")),   # New

        # ─── Central Network ───
        ("AGR", "JAI", EdgeAttr(5, 4, 0.994, "Fiber")),
        ("JAI", "IDR", EdgeAttr(7, 5, 0.995, "Fiber")),
        ("IDR", "BHO", EdgeAttr(4, 6, 0.98, "Cable")),
        ("BHO", "KNP", EdgeAttr(8, 12, 0.97, "Cable")),
        ("JAI", "BHO", EdgeAttr(8, 13, 0.97, "Cable")),
        ("BHO", "NAG", EdgeAttr(6, 8, 0.985, "Cable")),
        ("IDR", "AMD", EdgeAttr(7, 5, 0.995, "Fiber")),
        ("NAG", "RPR", EdgeAttr(5, 4, 0.998, "Fiber")),
        ("RPR", "BBI", EdgeAttr(8, 6, 0.993, "Fiber")),
        ("IDR", "NAG", EdgeAttr(6, 7, 0.995, "Fiber")),     # New
        ("BHO", "RPR", EdgeAttr(9, 10, 0.98, "Cable")),     # New

        # ─── Western Network ───
        ("AMD", "STV", EdgeAttr(4, 3, 0.998, "Fiber")),
        ("STV", "MUM", EdgeAttr(5, 4, 0.999, "Fiber")),
        ("MUM", "PUN", EdgeAttr(3, 3, 0.999, "Fiber")),
        ("MUM", "NAG", EdgeAttr(10, 14, 0.975, "Cable")),
        ("PUN", "NAG", EdgeAttr(11, 15, 0.970, "Cable")),
        ("PUN", "GOA", EdgeAttr(7, 10, 0.98, "Cable")),
        ("MUM", "GOA", EdgeAttr(8, 11, 0.975, "Cable")),
        ("AMD", "MUM", EdgeAttr(6, 5, 0.998, "Fiber")),     # New
        ("STV", "IDR", EdgeAttr(7, 8, 0.985, "Cable")),     # New

        # ─── Southern & Coastal Network ───
        ("MUM", "HYD", EdgeAttr(12, 16, 0.97, "Cable")),
        ("NAG", "HYD", EdgeAttr(8, 10, 0.985, "Cable")),
        ("HYD", "VIZ", EdgeAttr(9, 12, 0.978, "Cable")),
        ("RPR", "VIZ", EdgeAttr(7, 9, 0.98, "Cable")),
        ("BBI", "VIZ", EdgeAttr(6, 5, 0.995, "Fiber")),
        ("VIZ", "CHN", EdgeAttr(10, 7, 0.996, "Fiber")),
        ("HYD", "BLR", EdgeAttr(8, 5, 0.997, "Fiber")),
        ("HYD", "CHN", EdgeAttr(9, 13, 0.975, "Cable")),
        ("PUN", "BLR", EdgeAttr(14, 8, 0.995, "Fiber")),
        ("GOA", "BLR", EdgeAttr(9, 12, 0.98, "Cable")),
        ("GOA", "CJB", EdgeAttr(10, 14, 0.975, "Cable")),
        ("BLR", "CHN", EdgeAttr(6, 4, 0.998, "Fiber")),
        ("BLR", "CJB", EdgeAttr(5, 4, 0.997, "Fiber")),
        ("CHN", "CJB", EdgeAttr(7, 9, 0.985, "Cable")),
        ("CJB", "KOC", EdgeAttr(4, 3, 0.998, "Fiber")),
        ("KOC", "TRV", EdgeAttr(4, 3, 0.999, "Fiber")),
        ("KOC", "IXM", EdgeAttr(5, 6, 0.98, "Cable")),
        ("TRV", "IXM", EdgeAttr(4, 4, 0.998, "Fiber")),
        ("IXM", "CHN", EdgeAttr(8, 5, 0.996, "Fiber")),
        ("MUM", "BLR", EdgeAttr(15, 9, 0.996, "Fiber")),    # New
        ("HYD", "CJB", EdgeAttr(11, 13, 0.98, "Cable")),    # New

        # ─── SATELLITE SHORTCUTS (The Traps) ───
        ("DEL", "CHN", EdgeAttr(3, 140, 0.85, "Satellite")),
        ("DEL", "BLR", EdgeAttr(4, 130, 0.86, "Satellite")),
        ("DEL", "MUM", EdgeAttr(2, 110, 0.88, "Satellite")),
        ("DEL", "KOL", EdgeAttr(3, 120, 0.87, "Satellite")),
        ("MUM", "KOL", EdgeAttr(3, 135, 0.84, "Satellite")),
        ("MUM", "CHN", EdgeAttr(2, 125, 0.85, "Satellite")),
        ("KOL", "CHN", EdgeAttr(3, 130, 0.86, "Satellite")),
        ("DEL", "GAU", EdgeAttr(2, 125, 0.84, "Satellite")),
        ("BLR", "KOL", EdgeAttr(3, 135, 0.85, "Satellite")),
        ("SXR", "TRV", EdgeAttr(5, 180, 0.80, "Satellite")), # Extreme distance
        ("AMD", "GAU", EdgeAttr(4, 150, 0.82, "Satellite")),
        ("IXC", "CJB", EdgeAttr(4, 160, 0.81, "Satellite")), # New TRAP!
    ]

    for u, v, attr in edges:
        g.add_edge(u, v, attr, bidirectional=True)

    return g


def get_all_edges(graph):
    """Return a list of (u, v, EdgeAttr) for all unique undirected edges."""
    seen = set()
    edges = []
    for u in sorted(graph.nodes):
        for v, attr in graph.neighbors(u):
            key = tuple(sorted([u, v]))
            if key not in seen:
                seen.add(key)
                edges.append((u, v, attr))
    return edges


def get_top_routes(graph, src, dst, k=10):
    """Enumerate top-k routes by cost using Yen's algorithm."""
    G = nx.DiGraph()
    for u in graph.nodes:
        G.add_node(u)
    for u, nbrs in graph.adj.items():
        for v, a in nbrs:
            G.add_edge(u, v, weight=a.cost)
    routes = []
    try:
        for path in nx.shortest_simple_paths(G, src, dst, weight='weight'):
            m = graph.path_metrics(list(path), src, dst)
            routes.append(m)
            if len(routes) >= k:
                break
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass
    return routes


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DIJKSTRA BASELINES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _dijkstra(graph, src, dst, weight_fn):
    dist = {n: float('inf') for n in graph.nodes}
    dist[src] = 0.0
    prev = {}
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == dst:
            break
        for v, a in graph.neighbors(u):
            nd = d + weight_fn(a)
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if dist[dst] == float('inf'):
        return [], float('inf')
    path, node = [], dst
    while node != src:
        path.append(node)
        node = prev[node]
    path.append(src)
    return path[::-1], dist[dst]


def dijkstra_cost(graph, src, dst):
    return _dijkstra(graph, src, dst, lambda a: a.cost)


def dijkstra_latency(graph, src, dst):
    return _dijkstra(graph, src, dst, lambda a: a.latency)


def dijkstra_reliability(graph, src, dst):
    path, _ = _dijkstra(graph, src, dst,
                        lambda a: -math.log(max(a.reliability, 1e-9)))
    return path, graph.path_reliability(path) if path else 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GENETIC ALGORITHM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _random_path(graph, src, dst, max_depth=15, max_attempts=100):
    """Stochastic DFS with strict depth cap."""
    for _ in range(max_attempts):
        path = [src]
        visited = {src}
        node = src
        for _ in range(max_depth):
            if node == dst:
                return path
            nbrs = [v for v, _ in graph.neighbors(node) if v not in visited]
            if not nbrs:
                break
            node = random.choice(nbrs)
            path.append(node)
            visited.add(node)
    return None


def _init_population(graph, src, dst, size):
    """Seed with Yen's top-3 shortest + random walks for diversity."""
    pop = []
    G = nx.DiGraph()
    for u in graph.nodes:
        G.add_node(u)
    for u, nbrs in graph.adj.items():
        for v, a in nbrs:
            G.add_edge(u, v, weight=a.cost)
    try:
        for i, path in enumerate(nx.shortest_simple_paths(G, src, dst, weight='weight')):
            pop.append(list(path))
            if i >= 2:
                break
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass

    attempts = 0
    while len(pop) < size and attempts < size * 10:
        p = _random_path(graph, src, dst)
        if p and graph.is_valid_path(p, src, dst):
            pop.append(p)
        attempts += 1
    return pop


def _eval_path(graph, path, src, dst, cfg):
    if not graph.is_valid_path(path, src, dst):
        return (float('inf'), float('inf'), float('inf')), float('inf')
    
    cost = graph.path_cost(path)
    lat = graph.path_latency(path)
    rel = graph.path_reliability(path)
    
    cv = 0.0
    if lat > cfg.max_latency:
        cv += (lat - cfg.max_latency) / max(1.0, cfg.max_latency)
    if rel < cfg.min_reliability:
        cv += (cfg.min_reliability - rel)
        
    objs = (cost, lat, 1.0 - rel)
    return objs, cv


def _dominates(a_objs, a_cv, b_objs, b_cv):
    """Constrained Domination: returns True if a dominates b."""
    if a_cv == 0 and b_cv > 0:
        return True
    if a_cv > 0 and b_cv == 0:
        return False
    if a_cv > 0 and b_cv > 0:
        return a_cv < b_cv
    
    better_in_any = False
    for ao, bo in zip(a_objs, b_objs):
        if ao > bo:
            return False
        if ao < bo:
            better_in_any = True
    return better_in_any


def _fast_non_dominated_sort(pop_objs, pop_cvs):
    fronts = [[]]
    domination_count = [0] * len(pop_objs)
    dominated_solutions = [[] for _ in range(len(pop_objs))]
    rank = [0] * len(pop_objs)
    
    for p in range(len(pop_objs)):
        for q in range(len(pop_objs)):
            if p == q: continue
            if _dominates(pop_objs[p], pop_cvs[p], pop_objs[q], pop_cvs[q]):
                dominated_solutions[p].append(q)
            elif _dominates(pop_objs[q], pop_cvs[q], pop_objs[p], pop_cvs[p]):
                domination_count[p] += 1
                
        if domination_count[p] == 0:
            rank[p] = 0
            fronts[0].append(p)
            
    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
        
    return fronts[:-1], rank


def _crowding_distance(front, pop_objs):
    distance = {i: 0.0 for i in front}
    if len(front) == 0:
        return distance
    if len(front) <= 2:
        for i in front:
            distance[i] = float('inf')
        return distance

    for m in range(3): # 3 objectives
        front_sorted = sorted(front, key=lambda i: pop_objs[i][m])
        distance[front_sorted[0]] = float('inf')
        distance[front_sorted[-1]] = float('inf')
        m_min = pop_objs[front_sorted[0]][m]
        m_max = pop_objs[front_sorted[-1]][m]
        
        if m_max - m_min == 0:
            continue
            
        for i in range(1, len(front_sorted) - 1):
            distance[front_sorted[i]] += (pop_objs[front_sorted[i+1]][m] - pop_objs[front_sorted[i-1]][m]) / (m_max - m_min)
            
    return distance


def _tournament_nsga2(pop_objs, pop_cvs, ranks, distances, k=4):
    contenders = random.sample(range(len(pop_objs)), min(k, len(pop_objs)))
    best = contenders[0]
    for i in contenders[1:]:
        if pop_cvs[i] == 0 and pop_cvs[best] > 0:
            best = i
        elif pop_cvs[i] > 0 and pop_cvs[best] == 0:
            pass
        elif pop_cvs[i] > 0 and pop_cvs[best] > 0:
            if pop_cvs[i] < pop_cvs[best]:
                best = i
        else:
            if ranks[i] < ranks[best]:
                best = i
            elif ranks[i] == ranks[best]:
                if distances.get(i, 0) > distances.get(best, 0):
                    best = i
    return best


def _crossover(graph, pa, pb):
    """Shared-node crossover with repair fallback."""
    common = [n for n in pb[1:-1] if n in set(pa[1:-1])]
    if common:
        node = random.choice(common)
        child = pa[:pa.index(node) + 1] + pb[pb.index(node) + 1:]
        if len(child) == len(set(child)) and graph.is_valid_path(child, pa[0], pa[-1]):
            return child
    if len(pa) > 2 and len(pb) > 2:
        mid = random.randint(1, len(pa) - 1)
        bridge = _random_path(graph, pa[mid], pb[-1], max_attempts=15)
        if bridge and len(bridge) > 1:
            child = pa[:mid] + bridge
            if len(child) == len(set(child)) and graph.is_valid_path(child, pa[0], pa[-1]):
                return child
    return None


def _mutate(graph, path, src, dst):
    """Replace a random suffix with a new random walk."""
    if len(path) <= 2:
        return path
    cut = random.randint(1, len(path) - 2)
    tail = _random_path(graph, path[cut], dst, max_attempts=30)
    if tail:
        new = path[:cut] + tail
        if len(new) == len(set(new)) and graph.is_valid_path(new, src, dst):
            return new
    return path


def _pop_diversity(population):
    return len({tuple(p) for p in population}) / max(len(population), 1)


def run_ga(graph, src, dst, cfg=None):
    """
    Run NSGA-II Genetic Algorithm and return best path based on preferences.
    """
    if cfg is None:
        cfg = GAConfig()
    t0 = time.time()

    population = _init_population(graph, src, dst, cfg.pop_size)
    if not population:
        return GAResult([], float('inf'), float('inf'), 0.0, 0.0, [], False)

    snapshots = []
    best_path = None
    best_cost = float('inf')

    def evaluate_pop(pop):
        objs, cvs = [], []
        for p in pop:
            o, c = _eval_path(graph, p, src, dst, cfg)
            objs.append(o)
            cvs.append(c)
        return objs, cvs

    objs, cvs = evaluate_pop(population)

    for gen in range(cfg.max_gen):
        progress = gen / max(cfg.max_gen, 1)
        mut_rate = cfg.mut_rate_init + progress * (cfg.mut_rate_min - cfg.mut_rate_init)

        fronts, ranks = _fast_non_dominated_sort(objs, cvs)
        distances = {}
        for f in fronts:
            f_dist = _crowding_distance(f, objs)
            distances.update(f_dist)

        valid_indices = [i for i, cv in enumerate(cvs) if cv == 0]
        gen_best_cost = float('inf')
        gen_best_path = None
        
        if valid_indices:
            best_idx = valid_indices[0]
            best_score = float('inf')
            for i in valid_indices:
                if cfg.cost_only:
                    sc = objs[i][0]
                else:
                    sc = cfg.alpha * objs[i][0] + cfg.beta * objs[i][1] + cfg.gamma * objs[i][2]
                if sc < best_score:
                    best_score = sc
                    best_idx = i
            
            gen_best_cost = objs[best_idx][0]
            gen_best_path = population[best_idx]
            
            if gen_best_cost < best_cost:
                best_cost = gen_best_cost
                best_path = copy.deepcopy(gen_best_path)

        div = _pop_diversity(population)
        valid_costs = [objs[i][0] for i in valid_indices]
        avg_cost = float(np.mean(valid_costs)) if valid_costs else float('inf')

        snapshots.append(GenSnapshot(
            gen=gen, best_cost=best_cost, avg_cost=avg_cost,
            best_path=copy.deepcopy(best_path) if best_path else [],
            diversity=div, wall_time=time.time() - t0,
        ))

        offspring = []
        while len(offspring) < cfg.pop_size:
            pa_idx = _tournament_nsga2(objs, cvs, ranks, distances, cfg.tournament_k)
            pb_idx = _tournament_nsga2(objs, cvs, ranks, distances, cfg.tournament_k)
            pa = population[pa_idx]
            pb = population[pb_idx]
            
            child = None
            if random.random() < cfg.cx_rate:
                child = _crossover(graph, pa, pb)
            if child is None:
                child = copy.deepcopy(pa)
            if random.random() < mut_rate:
                child = _mutate(graph, child, src, dst)
            offspring.append(child)

        if div < cfg.diversity_threshold:
            n_inject = int(cfg.pop_size * cfg.restart_fraction)
            for _ in range(n_inject):
                p = _random_path(graph, src, dst)
                if p and graph.is_valid_path(p, src, dst):
                    offspring.append(p)

        off_objs, off_cvs = evaluate_pop(offspring)
        
        combined_pop = population + offspring
        combined_objs = objs + off_objs
        combined_cvs = cvs + off_cvs
        
        c_fronts, c_ranks = _fast_non_dominated_sort(combined_objs, combined_cvs)
        
        new_pop = []
        new_objs = []
        new_cvs = []
        
        for f in c_fronts:
            if len(new_pop) + len(f) <= cfg.pop_size:
                for i in f:
                    new_pop.append(combined_pop[i])
                    new_objs.append(combined_objs[i])
                    new_cvs.append(combined_cvs[i])
            else:
                f_dist = _crowding_distance(f, combined_objs)
                sorted_f = sorted(f, key=lambda i: f_dist[i], reverse=True)
                for i in sorted_f:
                    if len(new_pop) < cfg.pop_size:
                        new_pop.append(combined_pop[i])
                        new_objs.append(combined_objs[i])
                        new_cvs.append(combined_cvs[i])
                    else:
                        break
                break
                
        population = new_pop
        objs = new_objs
        cvs = new_cvs

    feasible = best_cost < float('inf')
    return GAResult(
        best_path=best_path if best_path else [],
        best_cost=best_cost,
        best_latency=graph.path_latency(best_path) if best_path else float('inf'),
        best_reliability=graph.path_reliability(best_path) if best_path else 0.0,
        time_taken=time.time() - t0,
        generations=snapshots,
        feasible=feasible,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHALLENGE SCENARIOS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHALLENGES = {
    "🏢 Enterprise SLA": {
        "src": "DEL", "dst": "CHN",
        "max_latency": 50.0, "min_reliability": 0.0,
        "description": (
            "**Scenario:** Your enterprise SLA guarantees packet delivery under 50ms.\n\n"
            "Dijkstra's cheapest path uses a satellite shortcut with **130ms latency** — "
            "a massive SLA violation! The GA must find the cheapest *valid* route."
        ),
    },
    "🛡️ High Availability": {
        "src": "DEL", "dst": "CHN",
        "max_latency": 999.0, "min_reliability": 0.90,
        "description": (
            "**Scenario:** A UPI payment network requires ≥90% end-to-end reliability.\n\n"
            "Cheap cable and satellite links have poor reliability (84-96%). "
            "The GA must route around them while minimizing cost."
        ),
    },
    "🔒 Critical Infrastructure": {
        "src": "DEL", "dst": "CHN",
        "max_latency": 50.0, "min_reliability": 0.90,
        "description": (
            "**Scenario:** BOTH constraints active — latency ≤50ms AND reliability ≥90%.\n\n"
            "This is the hardest case. Most routes fail at least one constraint. "
            "Only the GA can navigate this constrained search space efficiently."
        ),
    },
    "📊 No Constraints (Baseline)": {
        "src": "DEL", "dst": "CHN",
        "max_latency": 999.0, "min_reliability": 0.0,
        "description": (
            "**Scenario:** No QoS constraints — pure cost minimization.\n\n"
            "Both algorithms should find the absolute cheapest path. "
            "Expected result: **TIE** — proving the GA matches Dijkstra's optimality."
        ),
    },
}
