import numpy as np
import math
from collections import defaultdict

def load_golden_gvrp_instance(file_path: str):

    with open(file_path, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    header = {}
    for ln in lines:
        up = ln.upper()
        if up == 'NODE_COORD_SECTION':
            break
        if ':' in ln:
            key, val = ln.split(':', 1)
        else:
            parts = ln.split(None, 1)
            if len(parts) != 2:
                continue
            key, val = parts
        header[key.strip().upper()] = val.strip()

    N = int(header['DIMENSION'])
    capacity = int(header['CAPACITY'])
    vehicles = int(header.get('VEHICLES', 0))

    coords = {}
    idx = lines.index('NODE_COORD_SECTION') + 1
    for i in range(idx, idx + N):
        nid, x, y = lines[i].split()
        coords[int(nid)] = (float(x), float(y))

    clusters_map = {}
    idx = lines.index('GVRP_SET_SECTION') + 1
    while idx < len(lines) and lines[idx].split()[0].isdigit():
        parts = list(map(int, lines[idx].split()))
        set_id = parts[0]
        for nid in parts[1:-1]:
            clusters_map[nid] = set_id
        idx += 1
    clusters_map[1] = 0

    cluster_demands = {}
    idx_d = lines.index('DEMAND_SECTION') + 1
    while idx_d < len(lines):
        parts = lines[idx_d].split()
        if len(parts) != 2 or not parts[0].isdigit():
            break
        cid, dem = map(int, parts)
        cluster_demands[cid] = dem
        idx_d += 1

    customers = []
    for orig_id in range(1, N+1):
        x, y = coords[orig_id]
        d = 1
        cl = clusters_map.get(orig_id, None)
        customers.append(Customer(orig_id-1, x, y, d, cluster=cl))

    depot = next((c for c in customers if c.cluster == 0), None)
    if depot is None:
        raise ValueError("No depot (cluster=0) found")

    clusters_dict = {}
    for nid, cl in clusters_map.items():
        if cl != 0:
            clusters_dict.setdefault(cl, []).append(nid)
    clusters = list(clusters_dict.values())

    N = len(customers)
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        xi, yi = coords[i+1]
        for j in range(N):
            xj, yj = coords[j+1]
            D[i, j] = math.hypot(xi - xj, yi - yj)

    return customers, depot, clusters, capacity, D, vehicles, cluster_demands

class Customer:
    def __init__(self, customer_id, x, y, demand, cluster=None):
        self.id = customer_id
        self.x = x
        self.y = y
        self.demand = demand
        self.cluster = cluster
        self.cluster_route = None

class Cluster:

    def __init__(self, cluster_id, nodes):
        self.id = cluster_id
        self.nodes = nodes
        self.demand = sum(n.demand for n in nodes)
        xs = [n.x for n in nodes]
        ys = [n.y for n in nodes]
        self.centroid = (sum(xs) / len(xs), sum(ys) / len(ys))

class VRPInstance:
    def __init__(
        self,
        customers,
        depot,
        vehicle_capacity,
        distance_matrix=None,
        max_vehicles=None,
        cluster_lists=None,
        cluster_demands=None
    ):

        self.customers = customers
        self.depot = depot
        self.vehicle_capacity = vehicle_capacity
        self.max_vehicles = max_vehicles
        self.cluster_lists = cluster_lists or []
        self.cluster_demands = cluster_demands or {}

        self.distance_matrix = (
            distance_matrix if distance_matrix is not None
            else self._build_distance_matrix()
        )
        self._build_clusters()

    @classmethod
    def from_golden_gvrp_file(cls, file_path: str):

        customers, depot, clusters, capacity, mat, vehicles, demands = \
            load_golden_gvrp_instance(file_path)
        return cls(
            customers=customers,
            depot=depot,
            vehicle_capacity=capacity,
            distance_matrix=mat,
            max_vehicles=vehicles,
            cluster_lists=clusters,
            cluster_demands=demands
        )

    def _build_distance_matrix(self):
        n = len(self.customers)
        dm = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = self.customers[i].x - self.customers[j].x
                dy = self.customers[i].y - self.customers[j].y
                dm[i, j] = np.hypot(dx, dy)
        return dm

    def _build_clusters(self):
        groups = defaultdict(list)
        for c in self.customers:
            if c.cluster is not None:
                groups[c.cluster].append(c)

        self.clusters = {cid: Cluster(cid, nodes)
                         for cid, nodes in groups.items()}
        self.n_clusters = len(self.clusters)

        self.cluster_distance_matrix = np.zeros((self.n_clusters, self.n_clusters))
        cids = sorted(self.clusters)
        for i, cid_i in enumerate(cids):
            xi, yi = self.clusters[cid_i].centroid
            for j, cid_j in enumerate(cids):
                xj, yj = self.clusters[cid_j].centroid
                self.cluster_distance_matrix[i, j] = np.hypot(xi - xj, yi - yj)

    def total_route_cost(self, route):
        cost = 0.0
        for i in range(len(route) - 1):
            cost += self.distance_matrix[route[i]][route[i+1]]
        return cost

    def is_feasible(self, route):

        total = sum(
            self.customers[idx].demand
            for idx in route
            if self.customers[idx].id != self.depot.id
        )
        return total <= self.vehicle_capacity

    def __repr__(self):
        return (
            f"VRPInstance(n_customers={len(self.customers)}, "
            f"n_clusters={self.n_clusters}, "
            f"capacity={self.vehicle_capacity}, "
            f"max_vehicles={self.max_vehicles})"
        )
