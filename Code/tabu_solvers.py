import random
import copy
from typing import List, Dict, Any, Tuple
import numpy as np
from local_search import LocalSearch


def compute_cost(routes: List[List[int]], D: np.ndarray) -> float:
    # Total length of all routes.
    return sum(D[r[i], r[i+1]] for r in routes for i in range(len(r)-1))


def compute_feasibility(routes: List[List[int]],
                        customers: List[Any],
                        demands: Dict[int, float],
                        capacity: int) -> bool:
    # Check that no route exceeds vehicle capacity.
    for r in routes:
        visited_clusters = {customers[idx].cluster for idx in r if idx != 0}
        load = sum(demands[c] for c in visited_clusters)
        if load > capacity:
            return False
    return True


def compute_route_demands(routes: List[List[int]],
                          customers: List[Any],
                          demands: Dict[int, float]) -> List[float]:
    # Compute total demand served by each route.
    route_loads = []
    for r in routes:
        visited = {customers[idx].cluster for idx in r if idx != 0}
        route_loads.append(sum(demands[c] for c in visited))
    return route_loads


class ClassicTabu:
    # Tabu search using intra-route 2-opt and swap neighborhoods.
    FIXED_TENURE = 10

    def __init__(
        self,
        D: np.ndarray,
        customers: List[Any],
        depot: int,
        capacity: int,
        cluster_demands: Dict[int, float],
        max_vehicles: int,
        iteration_limit: int = 500,
        candidate_list_size: int = 30
    ):
        self.D = D
        self.customers = customers
        self.depot = depot
        self.capacity = capacity
        self.demands = cluster_demands
        self.max_vehicles = max_vehicles
        self.iter_limit = iteration_limit
        self.cand_size = candidate_list_size
        self.tabu_list: Dict[Any, int] = {}

    def is_tabu(
        self,
        move: Any,
        iteration: int,
        cand_cost: float,
        best_cost: float
    ) -> bool:
        # Check if move is tabu, respecting aspiration.
        expire = self.tabu_list.get(move, -1)
        if iteration < expire and cand_cost >= best_cost:
            return True
        return False

    def set_tabu(self,
                 move: Any,
                 iteration: int):
        # Forbid move for FIXED_TENURE iterations.
        self.tabu_list[move] = iteration + ClassicTabu.FIXED_TENURE

    def two_opt_neighborhood(
        self,
        routes: List[List[int]]
    ) -> List[Tuple[List[List[int]], Any, float]]:
        # Generate intra-route 2-opt candidates.
        cands = []
        for ridx, route in enumerate(routes):
            n = len(route)
            if n < 5:
                continue
            for _ in range(self.cand_size):
                i, j = sorted(random.sample(range(1, n-1), 2))
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                new_routes = copy.deepcopy(routes)
                new_routes[ridx] = new_route
                if not compute_feasibility(new_routes,
                                           self.customers,
                                           self.demands,
                                           self.capacity):
                    continue
                cost = compute_cost(new_routes, self.D)
                move = (ridx, i, j)
                cands.append((new_routes, move, cost))
        return cands

    def swap_neighborhood(
        self,
        routes: List[List[int]]
    ) -> List[Tuple[List[List[int]], Any, float]]:
        # Generate intra-route swap candidates.
        cands = []
        for ridx, route in enumerate(routes):
            n = len(route)
            if n < 4:
                continue
            for _ in range(self.cand_size):
                i, j = random.sample(range(1, n-1), 2)
                new_route = route.copy()
                new_route[i], new_route[j] = new_route[j], new_route[i]
                new_routes = copy.deepcopy(routes)
                new_routes[ridx] = new_route
                if not compute_feasibility(new_routes,
                                           self.customers,
                                           self.demands,
                                           self.capacity):
                    continue
                cost = compute_cost(new_routes, self.D)
                move = (ridx, route[i], route[j])
                cands.append((new_routes, move, cost))
        return cands

    def search(
        self,
        initial_routes: List[List[int]],
        verbose: bool = False
    ) -> Tuple[List[List[int]], float, float, List[float]]:
        curr = copy.deepcopy(initial_routes)
        curr_cost = compute_cost(curr, self.D)
        best = copy.deepcopy(curr)
        best_cost = curr_cost
        self.tabu_list.clear()

        for it in range(1, self.iter_limit + 1):
            if verbose and it % 50 == 0:
                print(f"[Classic] iter {it:4d}, best={best_cost:.2f}")

            # combine neighborhoods
            c1 = self.two_opt_neighborhood(curr)
            c2 = self.swap_neighborhood(curr)
            candidates = c1 + c2
            if not candidates:
                break
            candidates.sort(key=lambda x: x[2])

            moved = False
            for sol, mv, cst in candidates:
                if not self.is_tabu(mv, it, cst, best_cost):
                    curr, curr_cost = sol, cst
                    self.set_tabu(mv, it)
                    moved = True
                    if curr_cost < best_cost:
                        best, best_cost = copy.deepcopy(curr), curr_cost
                    break
            if not moved:
                break

        loads = compute_route_demands(best, self.customers, self.demands)
        return best, best_cost, 0.0, loads


class AdaptiveTabu(ClassicTabu):
    # Adaptive tabu with intra-route 2-opt and swap neighborhoods.
    def __init__(
        self,
        D: np.ndarray,
        customers: List[Any],
        depot: int,
        capacity: int,
        cluster_demands: Dict[int, float],
        max_vehicles: int,
        iteration_limit: int = 500,
        candidate_list_size: int = 30,
        base_tenure: int = 5,
        intensify_limit: int = 5,
        diversify_limit: int = 20,
        initial_aspiration_relax: float = 0.05,
        max_aspiration_relax: float = 0.20
    ):
        super().__init__(D, customers, depot, capacity,
                         cluster_demands, max_vehicles,
                         iteration_limit, candidate_list_size)
        self.base_tenure = base_tenure
        self.intensify_limit = intensify_limit
        self.diversify_limit = diversify_limit
        self.min_cand_size = candidate_list_size
        self.max_cand_size = 2 * candidate_list_size
        self.curr_cand_size = candidate_list_size
        self.aspiration_relax = initial_aspiration_relax
        self.max_aspiration_relax = max_aspiration_relax
        self.last_improve_iter = 0

    def search(
        self,
        initial_routes: List[List[int]],
        verbose: bool = False
    ) -> Tuple[List[List[int]], float, float, List[float]]:
        curr = copy.deepcopy(initial_routes)
        curr_cost = compute_cost(curr, self.D)
        best = copy.deepcopy(curr)
        best_cost = curr_cost
        self.tabu_list.clear()
        self.last_improve_iter = 0

        for it in range(1, self.iter_limit + 1):
            if verbose and it % 5 == 0:
                stagn = it - self.last_improve_iter
                print(f"[Adaptive] it={it:3d} stagn={stagn:2d} cand={self.curr_cand_size} "
                      f"asp={self.aspiration_relax:.3f} best={best_cost:.2f}")

            self.cand_size = self.curr_cand_size
            # combine neighborhoods
            c1 = self.two_opt_neighborhood(curr)
            c2 = self.swap_neighborhood(curr)
            candidates = c1 + c2
            if not candidates:
                break
            candidates.sort(key=lambda x: x[2])

            moved = False
            for sol, mv, cst in candidates:
                stagn = it - self.last_improve_iter
                if stagn < self.intensify_limit:
                    tenure = self.base_tenure
                elif stagn < self.diversify_limit:
                    tenure = self.base_tenure + (stagn - self.intensify_limit)
                else:
                    tenure = max(1, self.base_tenure - (stagn - self.diversify_limit))

                expire = self.tabu_list.get(mv, -1)
                allow = (it >= expire) or (cst < best_cost * (1 + self.aspiration_relax))
                if allow:
                    curr, curr_cost = sol, cst
                    self.tabu_list[mv] = it + tenure

                    if curr_cost < best_cost:
                        best, best_cost = copy.deepcopy(curr), curr_cost
                        self.last_improve_iter = it
                        self.curr_cand_size = max(self.min_cand_size, self.curr_cand_size - 1)
                        self.aspiration_relax = min(self.max_aspiration_relax, self.aspiration_relax + 0.02)
                    else:
                        self.curr_cand_size = min(self.max_cand_size, self.curr_cand_size + 1)
                        self.aspiration_relax = max(0.01, self.aspiration_relax - 0.005)

                    moved = True
                    break
            if not moved:
                break

        loads = compute_route_demands(best, self.customers, self.demands)
        return best, best_cost, 0.0, loads
