import os
import matplotlib.pyplot as plt

def draw(routes, nodes, iteration, save_dir=None):
    plt.figure(figsize=(8, 6))
    for idx, route in enumerate(routes, 1):
        xs = [nodes[n].x for n in route]
        ys = [nodes[n].y for n in route]
        plt.plot(xs, ys, '-o', label=f'Route {idx}')

    depot = nodes.get(0)
    if depot:
        plt.scatter(depot.x, depot.y, marker='s', s=100, color='k', label='Depot')

    plt.title(f'Routes at iteration {iteration}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(loc='best', fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.5)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f"routes_{iteration}.png")
        plt.savefig(file_path)
    else:
        plt.savefig(f"routes_{iteration}.png")
    plt.close()

def plot_clusters(customers, clusters, save_path):
    plt.figure(figsize=(10, 8))
    colors = plt.get_cmap('tab20', len(clusters))
    for idx, cluster_nodes in enumerate(clusters):
        xs = [customers[n].x for n in cluster_nodes]
        ys = [customers[n].y for n in cluster_nodes]
        plt.scatter(xs, ys, color=colors(idx), label=f'Cluster {idx+1}')
    depot = next(c for c in customers if c is not None and c.cluster == 0)
    plt.scatter(depot.x, depot.y, color='k', marker='s', s=100, label='Depot')
    plt.title("Clusters")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend(loc='best', fontsize=6)
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_routes(customers, routes, save_path, title="Routes"):
    plt.figure(figsize=(10, 8))
    colors = plt.get_cmap('tab10', len(routes))
    for idx, route in enumerate(routes):
        xs = [customers[node].x for node in route]
        ys = [customers[node].y for node in route]
        plt.plot(xs, ys, marker='o', color=colors(idx), label=f'Route {idx+1}')
    depot = next(c for c in customers if c.cluster == 0)
    plt.scatter(depot.x, depot.y, color='k', marker='s', s=100, label='Depot')
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend(loc='best', fontsize=8)
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_best_instance(customers, clusters, initial_route, classic_route, adaptive_route, instance_id, results_folder):
    os.makedirs(results_folder, exist_ok=True)

    # Plot clusters
    cluster_plot_path = os.path.join(results_folder, f"clusters_{instance_id}.png")
    plot_clusters(customers, clusters, cluster_plot_path)

    # Plot initial solution
    initial_routes_plot_path = os.path.join(results_folder, f"initial_solution_{instance_id}.png")
    plot_routes(customers, initial_route, initial_routes_plot_path, "Initial Solution")

    # Plot Classic Tabu Solution
    classic_routes_plot_path = os.path.join(results_folder, f"classic_tabu_solution_{instance_id}.png")
    plot_routes(customers, classic_route, classic_routes_plot_path, "Classic Tabu Solution")

    # Plot Adaptive Tabu Solution
    adaptive_routes_plot_path = os.path.join(results_folder, f"adaptive_tabu_solution_{instance_id}.png")
    plot_routes(customers, adaptive_route, adaptive_routes_plot_path, "Adaptive Tabu Solution")
