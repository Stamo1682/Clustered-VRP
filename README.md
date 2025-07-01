# Clustered-VRP
Optimization Methodologies for the Clustered Vehicle Routing Problem

A Python implementation of the Golden VRP solver, combining clustering, TSP-based initial solutions, local search heuristics and classic and adaptive Tabu Search optimizers. This repository contains all code used for the experiments described in the accompanying thesis.

Repository Structure:

- data_loader.py – Load and parse GOLD-VRP instance files into VRPInstance objects.

- initial_solution.py – Generate initial feasible solutions using a TSP-splitting approach.

- local_search.py – Implement local search moves (2-opt, swap).

- tabu_solvers.py – Classic and Adaptive Tabu Search algorithms for improving routes.

- sol_drawer.py – Visualization utilities (Matplotlib) for plotting routes.

- results_analysis.py – Post-processing and plotting of results using Pandas and Plotnine.

- main.py – Driver script to run experiments on multiple instances and export results.

- Mesolora_2025.pdf – Thesis document detailing the methodology and findings.
