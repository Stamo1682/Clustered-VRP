import numpy as np
from typing import List, Any, Dict, Tuple

class LocalSearch:
    @staticmethod
    def _route_cost(r: List[int], D: np.ndarray) -> float:
        return sum(D[r[i], r[i+1]] for i in range(len(r)-1))

    @staticmethod
    def is_cluster_contiguous(route: List[int], customers: List[Any]) -> bool:
        clusters = {}
        for idx, node in enumerate(route):
            cl = customers[node].cluster
            if cl == 0:
                continue
            clusters.setdefault(cl, []).append(idx)
        for idxs in clusters.values():
            if max(idxs) - min(idxs) + 1 != len(idxs):
                return False
        return True

    @staticmethod
    def two_opt(r: List[int], D: np.ndarray) -> List[int]:
        best = r
        improved = True
        while improved:
            improved = False
            n = len(best)
            for i in range(1, n-2):
                for j in range(i+1, n-1):
                    cand = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    delta = LocalSearch._route_cost(cand, D) - LocalSearch._route_cost(best, D)
                    if delta < -1e-6:
                        best = cand
                        improved = True
                        break
                if improved:
                    break
        return best

    @staticmethod
    def swap_best(
        route: List[int], D: np.ndarray,
        customers: List[Any], capacity: float,
        demands: Dict[int, float]
    ) -> Tuple[List[int], float, int, int]:


        best_route = route
        best_delta = 0.0
        base = LocalSearch._route_cost(route, D)
        n = len(route)
        u = v = 0
        for i in range(1, n-2):
            for j in range(i+1, n-1):
                cand = route[:]
                cand[i], cand[j] = cand[j], cand[i]
                if not LocalSearch.is_cluster_contiguous(cand, customers):
                    continue

                new_cost = LocalSearch._route_cost(cand, D)
                delta = new_cost - base
                if delta < best_delta:
                    best_delta = delta
                    best_route = cand

                    k = min(i, j)
                    u, v = best_route[k-1], best_route[k]
        return best_route, best_delta, u, v

