import numpy as np
from typing import List, Dict, Tuple
from local_search import LocalSearch
from data_loader import Customer
import math

def nearest_neighbor_tour(centers: List[Customer], D: np.ndarray) -> List[int]:
    n = len(centers)
    unvisited = set(range(1, n))
    tour = [0]
    curr = 0

    # 1) Nearest‐neighbor to build a closed loop
    while unvisited:
        nxt = min(unvisited, key=lambda j: D[curr, j])
        unvisited.remove(nxt)
        tour.append(nxt)
        curr = nxt
    tour.append(0)

    # 2) 2-opt improvement on that loop
    improved = LocalSearch.two_opt(tour, D)
    return improved


def compute_route_cost(seq: List[int], D: np.ndarray) -> float:
    return sum(D[seq[i], seq[i+1]] for i in range(len(seq)-1))


class TSPSplitSolver:
    def __init__(
        self,
        customers: List[Customer],
        distance_matrix: np.ndarray,
        vehicle_capacity: float,
        max_vehicles: int,
        cluster_demands: Dict[int, float]
    ):
        self.customers = customers
        self.D_full = distance_matrix
        self.Q = vehicle_capacity
        self.V = max_vehicles
        self.demands = cluster_demands

    def _build_centroids(self) -> Tuple[List[Customer], Dict[int, List[int]], Dict[int, float]]:
        clusters: Dict[int, List[Customer]] = {}
        for c in self.customers:
            if c.cluster != 0:
                clusters.setdefault(c.cluster, []).append(c)

        centers: List[Customer] = [self.customers[0]]
        mapping: Dict[int, List[int]] = {}
        demand: Dict[int, float] = {}
        idx = 1

        for cl_id, nodes in sorted(clusters.items()):
            cx = sum(n.x for n in nodes) / len(nodes)
            cy = sum(n.y for n in nodes) / len(nodes)
            cent = Customer(idx, cx, cy, demand=0, cluster=cl_id)
            cent.cluster_nodes = nodes
            centers.append(cent)

            mapping[idx] = [n.id for n in nodes]
            demand[idx] = self.demands[cl_id]
            idx += 1

        return centers, mapping, demand

    def _build_centroid_dist(self, centers: List[Customer]) -> np.ndarray:
        n = len(centers)
        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i].x - centers[j].x
                dy = centers[i].y - centers[j].y
                d = math.hypot(dx, dy)
                D[i, j] = D[j, i] = d
        return D

    def _split_dp(self, tour, demand, D):
        M = len(tour)
        dp = [(float('inf'), -1)] * M
        dp[0] = (0.0, -1)

        for j in range(1, M):
            best = (float('inf'), -1)
            load = 0.0

            for i in range(j - 1, -1, -1):
                nxt_node = tour[i + 1]
                if nxt_node != 0:
                    load += demand.get(nxt_node, 0)
                if load > self.Q:
                    break

                sub = [0] + tour[i + 1:j + 1] + [0]
                cost_sub = compute_route_cost(sub, D)
                total = dp[i][0] + cost_sub
                if total < best[0]:
                    best = (total, i)

            dp[j] = best

        segments = []
        idx = M - 1
        while idx > 0:
            prev = dp[idx][1]
            seg = tour[prev + 1: idx + 1]
            segments.append(seg)
            idx = prev
        segments.reverse()
        return segments

    def solve(self) -> List[List[int]]:
        # Build centroids + demands
        centers, mapping, demand = self._build_centroids()

        # Build centroid‐level distance matrix
        D_cent = self._build_centroid_dist(centers)

        # TSP on centroids (NN + 2-opt inside the helper)
        tour2 = nearest_neighbor_tour(centers, D_cent)
        tour2 = tour2[:-1]  # remove trailing 0 for the split

        # Optimal split into ≤ V centroid‐routes
        cent_routes = self._split_dp(tour2, demand, D_cent)

        print("  [DEBUG] centroid-routes and their demands:")
        for idx, seg in enumerate(cent_routes, 1):
            seg_d = sum(demand[cidx] for cidx in seg)
            print(f"    Segment {idx}: clusters {seg} → demand {seg_d}")
        for seg in cent_routes:
            assert sum(demand[cidx] for cidx in seg) <= self.Q, \
                f"Capacity violated on segment {seg}"

        # 5) Expand each centroid‐route to real‐customer loops (without 2-opt refinement)
        full: List[List[int]] = []
        for cr in cent_routes:
            route = [0]
            for cidx in cr:
                ids = mapping[cidx]
                # Create simple loop without 2-opt
                loop = ids + [ids[0]]
                route += loop[:-1]
            route.append(0)
            full.append(route)

        return full


class TSPSplitInitialSolver:

    def __init__(
        self,
        customers: List[Customer],
        distance_matrix: np.ndarray,
        vehicle_capacity: float,
        max_vehicles: int,
        cluster_demands: Dict[int, float]
    ):
        self.customers = customers
        self.D_full = distance_matrix
        self.Q = vehicle_capacity
        self.V = max_vehicles
        self.cluster_demands = cluster_demands

    def solve(self) -> Tuple[List[List[int]], float]:
        splitter = TSPSplitSolver(
            self.customers,
            self.D_full,
            self.Q,
            self.V,
            self.cluster_demands
        )
        routes = splitter.solve()
        cost = sum(
            self.D_full[r[i], r[i+1]]
            for r in routes for i in range(len(r)-1)
        )
        return routes, cost
