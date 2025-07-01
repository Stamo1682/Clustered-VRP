import argparse
import shutil
import zipfile
import tempfile
import time
from pathlib import Path
import csv

from data_loader import VRPInstance
from initial_solution import TSPSplitInitialSolver
from tabu_solvers import ClassicTabu, AdaptiveTabu
from sol_drawer import plot_best_instance


def process_instance(
    inst_name: str,
    inst: VRPInstance,
    init_writer,
    tabu_writer,
    results_writer,
    iters: int,
    cand: int
):
    splitter = TSPSplitInitialSolver(
        customers=inst.customers,
        distance_matrix=inst.distance_matrix,
        vehicle_capacity=inst.vehicle_capacity,
        max_vehicles=inst.max_vehicles,
        cluster_demands=inst.cluster_demands
    )
    initial_routes, init_cost = splitter.solve()
    print(f"[{inst_name}] Initial cost: {init_cost:.2f}")

    for rid, route in enumerate(initial_routes, start=1):
        rcost = sum(
            inst.distance_matrix[route[i], route[i+1]]
            for i in range(len(route)-1)
        )
        for node in route:
            cust = inst.customers[node]
            init_writer.writerow([
                inst_name, rid, node, cust.x, cust.y, f"{rcost:.2f}"
            ])

    print(f"[{inst_name}] Starting Classic Tabu (iters={iters}, cand={cand})")
    t0 = time.perf_counter()
    classic_routes, cost_c, _, _ = ClassicTabu(
        D=inst.distance_matrix,
        customers=inst.customers,
        depot=inst.depot.id,
        capacity=inst.vehicle_capacity,
        cluster_demands=inst.cluster_demands,
        max_vehicles=inst.max_vehicles,
        iteration_limit=iters,
        candidate_list_size=cand
    ).search(initial_routes, verbose=True)
    t_classic = time.perf_counter() - t0
    print(f"[{inst_name}] Classic done: cost={cost_c:.2f}, time={t_classic:.2f}s")

    print(f"[{inst_name}] Starting Adaptive Tabu (iters={iters}, cand={cand})")
    t1 = time.perf_counter()
    adaptive_routes, cost_a, _, _ = AdaptiveTabu(
        D=inst.distance_matrix,
        customers=inst.customers,
        depot=inst.depot.id,
        capacity=inst.vehicle_capacity,
        cluster_demands=inst.cluster_demands,
        max_vehicles=inst.max_vehicles,
        iteration_limit=iters,
        candidate_list_size=cand,
        base_tenure=5,
        intensify_limit=5,
        diversify_limit=20,
        initial_aspiration_relax=0.05,
        max_aspiration_relax=0.20
    ).search(initial_routes, verbose=True)
    t_adaptive = time.perf_counter() - t1
    print(f"[{inst_name}] Adaptive done: cost={cost_a:.2f}, time={t_adaptive:.2f}s")

    num_routes = len(initial_routes)
    classic_rate = (init_cost - cost_c) / t_classic if t_classic > 0 else 0.0
    adaptive_rate = (init_cost - cost_a) / t_adaptive if t_adaptive > 0 else 0.0
    improvement_pct = (cost_c - cost_a) / cost_a * 100 if cost_a > 0 else 0.0

    tabu_writer.writerow([
        inst_name,
        inst.max_vehicles,
        num_routes,
        f"{cost_c:.2f}",
        f"{cost_a:.2f}",
        f"{t_classic:.2f}",
        f"{t_adaptive:.2f}",
        f"{classic_rate:.2f}",
        f"{adaptive_rate:.2f}",
        f"{improvement_pct:.2f}%"
    ])

    results_writer.writerow([
        inst_name,
        inst.max_vehicles,
        num_routes,
        f"{cost_c:.2f}",
        f"{cost_a:.2f}",
        f"{t_classic:.2f}",
        f"{t_adaptive:.2f}",
        f"{classic_rate:.2f}",
        f"{adaptive_rate:.2f}",
        f"{improvement_pct:.2f}%"
    ])

    return {
        'inst': inst,
        'initial_routes': initial_routes,
        'classic_routes': classic_routes,
        'adaptive_routes': adaptive_routes,
        'improvement': improvement_pct
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Tabu tests on .gvrp instances"
    )
    parser.add_argument(
        "--zip",
        dest="zip_path",
        default=r"C:\Users\fani_\OneDrive\Υπολογιστής\For GitHub\Golden.zip",
        help="Path to Golden.zip or extracted folder")
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--cand", type=int, default=30)
    args = parser.parse_args()

    DATA_ZIP_PATH = Path(args.zip_path)
    RESULTS_FOLDER = Path("Results")
    INITIAL_CSV = RESULTS_FOLDER / "initial_routes.csv"
    TABU_SUMMARY_CSV = RESULTS_FOLDER / "tabu_search_results.csv"
    RESULTS_CSV = RESULTS_FOLDER / "Tabu_Results.csv"

    if RESULTS_FOLDER.exists():
        shutil.rmtree(RESULTS_FOLDER)
    RESULTS_FOLDER.mkdir(parents=True)

    summary_data = []
    with open(INITIAL_CSV, "w", newline="") as init_f, \
         open(TABU_SUMMARY_CSV, "w", newline="") as tab_f, \
         open(RESULTS_CSV, "w", newline="") as res_f:

        init_writer = csv.writer(init_f)
        init_writer.writerow(["instance", "route_id", "node_id", "x", "y", "route_cost"])

        tabu_writer = csv.writer(tab_f)
        tabu_writer.writerow([
            "Instance", "Available Vehicles", "Number of Routes",
            "Classic Cost", "Adaptive Cost", "Classic Time", "Adaptive Time",
            "Classic Rate", "Adaptive Rate", "Improvement %"
        ])

        results_writer = csv.writer(res_f)
        results_writer.writerow([
            "Instance", "Available Vehicles", "Number of Routes",
            "Classic Cost", "Adaptive Cost", "Classic Time", "Adaptive Time",
            "Classic Rate", "Adaptive Rate", "Improvement %"
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            if DATA_ZIP_PATH.is_dir():
                base_dir = DATA_ZIP_PATH
            else:
                with zipfile.ZipFile(DATA_ZIP_PATH, 'r') as zf:
                    zf.extractall(tmpdir)
                base_dir = Path(tmpdir)

            gfiles = list(base_dir.rglob("*.gvrp"))
            print("Found GVRP instances:", [g.name for g in gfiles])

            for gfile in gfiles:
                inst_name = gfile.stem
                print(f"=== Processing {inst_name} ===")
                inst = VRPInstance.from_golden_gvrp_file(str(gfile))

                data = process_instance(
                    inst_name, inst,
                    init_writer, tabu_writer, results_writer,
                    args.iters, args.cand
                )
                summary_data.append(data)

    best = max(summary_data, key=lambda x: x['improvement'])
    inst = best['inst']
    print(f"Best improvement: {inst.depot.id} -> {best['improvement']:.2f}%")

    clusters_ids = [[c.cluster] for c in inst.customers if c.id != inst.depot.id]
    plot_dir = RESULTS_FOLDER / "Plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_best_instance(
        customers=inst.customers,
        clusters=clusters_ids,
        initial_route=best['initial_routes'],
        classic_route=best['classic_routes'],
        adaptive_route=best['adaptive_routes'],
        instance_id=inst.depot.id,
        results_folder=str(plot_dir)
    )

    print(f"Plotted best instance to {plot_dir}")
    print("Done. Results in:", RESULTS_FOLDER.absolute())


if __name__ == "__main__":
    main()
