import os
import pandas as pd
import shutil
from plotnine import (
    geom_line,
    coord_cartesian,
    scale_fill_manual,
    scale_color_manual,
    ggplot, aes, geom_col, theme_bw, theme,
    element_text, scale_y_continuous, labs
)

# Paths
SUMMARY_CSV = os.path.join("Results", "tabu_search_results.csv")

def prepare_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

df = pd.read_csv(SUMMARY_CSV)

print("Available columns:", df.columns.tolist())

def find_col(keywords):
    for col in df.columns:
        for kw in keywords:
            if kw.lower() in col.lower():
                return col
    return None

imp_col = find_col(["improvement %", "improvement"])
if imp_col is None:
    raise KeyError("Improvement column not found")
best_row = df.loc[df[imp_col].idxmax()]
best_label = best_row.get('Instance', str(best_row.name))

PLOTS_DIR = os.path.join("Results", f"analysis{best_label}")
prepare_dir(PLOTS_DIR)

axis_theme = theme(
    axis_text_x=element_text(rotation=90, hjust=1, size=8),
    axis_text_y=element_text(size=8)
)

# 1. Total cost comparison (Classic vs Adaptive)
cost_cols = [find_col(["classic cost"]), find_col(["adaptive cost"])]
if None in cost_cols:
    raise KeyError(f"Cost columns not found: {cost_cols}")
df_cost = (
    df.melt(
        id_vars=['Instance'],
        value_vars=cost_cols,
        var_name='Solver',
        value_name='Cost'
    )
)

p1 = (
    ggplot(df_cost, aes(x='Instance', y='Cost', color='Solver', group='Solver'))
    + geom_line(size=1)
    + scale_color_manual(values={
        cost_cols[0]: '#003366',
        cost_cols[1]: '#a31212'
    })
    + theme_bw(base_size=12)
    + axis_theme
    + theme(legend_position='bottom')
    + labs(
        title="Total Cost per Instance",
        x="Instance",
        y="Cost"
    )
)
p1.save(
    os.path.join(PLOTS_DIR, "total_costs.png"),
    width=12, height=4, units="in", dpi=300
)
print("Saved total_costs.png")

# 2. Improvement gap (Adaptive improvement over Classic %)

df['Improvement %'] = (
    df['Improvement %'].astype(str)
    .str.rstrip('%')
    .str.replace(',', '.', regex=False)
    .astype(float)
)

print("After conversion:", df['Improvement %'].dtype)
print(df['Improvement %'].head())

p2 = (
    ggplot(df, aes(x='Instance', y='Improvement %'))
    + geom_col(fill='#003366')
    + scale_y_continuous(
        labels=lambda breaks: [f"{b:.0f}" for b in breaks]
    )
    + theme_bw(base_size=12)
    + axis_theme
    + theme(legend_position='none')
    + labs(
        title="Adaptive Improvement over Classic (%)",
        x="Instance",
        y="Gap (%)"
    )
)
p2.save(
    os.path.join(PLOTS_DIR, "improvement_gap.png"),
    width=10, height=4, units="in", dpi=300
)
print("Saved improvement_gap.png")

# 3. CPU time comparison (Classic vs Adaptive)
time_cols = [find_col(["classic time", "classic cpu"]), find_col(["adaptive time", "adaptive cpu"])]
if None in time_cols:
    raise KeyError(f"Time columns not found: {time_cols}")
df_time = df.melt(id_vars=['Instance'], value_vars=time_cols, var_name='Solver', value_name='Seconds')

df_time['Solver'] = pd.Categorical(
    df_time['Solver'],
    categories=[time_cols[1], time_cols[0]],
    ordered=True
)

p3 = (
    ggplot(df_time, aes(x='Instance', y='Seconds', fill='Solver'))
    + geom_col()
    + scale_fill_manual(values={
        time_cols[1]: '#a31212',  # Adaptive Time
        time_cols[0]: '#003366'   # Classic Time
    })
    + coord_cartesian(ylim=(0, None))
    + theme_bw(base_size=12)
    + axis_theme
    + theme(legend_position='bottom')
    + labs(title="CPU Time per Instance", x="Instance", y="Seconds")
)
p3.save(
    os.path.join(PLOTS_DIR, "cpu_time_stacked.png"),
    width=12, height=4, units="in", dpi=300
)
print("Saved cpu_time_stacked.png")

# 4. Fleet size for adaptive solutions
route_col = find_col(["route", "numroutes"])
if route_col:
    p4 = (
        ggplot(df, aes(x='Instance', y=route_col))
        + geom_col(stat='identity', fill='#003366')
        + theme_bw(base_size=12)
        + axis_theme
        + theme(legend_position='none')
        + labs(title="Number of Routes in Adaptive Solution", x="Instance", y=route_col)
    )
    p4.save(
        os.path.join(PLOTS_DIR, "fleet_size.png"),
        width=10, height=4, units="in", dpi=300
    )
    print("Saved fleet_size.png")
else:
    print("Skipping fleet_size: no route-like column found.")

# 5. Improvement ratio distribution (Gap% per CPU-second)
imp_vals = df[imp_col].astype(str).str.rstrip('%').astype(float)
classic_time_col = time_cols[0]
df['Imp_Ratio'] = imp_vals / df[classic_time_col]

p5 = (
    ggplot(df, aes(x='Instance', y='Imp_Ratio'))
    + geom_col(stat='identity', fill='#003366')
    + theme_bw(base_size=12)
    + axis_theme
    + theme(legend_position='none')
    + labs(title="Improvement Ratio (Gap% per CPU-second)", x="Instance", y="Improvement Ratio")
)
p5.save(
    os.path.join(PLOTS_DIR, "improvement_ratio.png"),
    width=10, height=4, units="in", dpi=300
)
print("Saved improvement_ratio.png")

# 6. Copy best instance solution plots
BEST_DIR = os.path.join("Results", "Analysis")
best_dest = os.path.join(PLOTS_DIR, f"best_{best_label}")
if os.path.isdir(BEST_DIR):
    shutil.copytree(BEST_DIR, best_dest)
    print(f"Copied best instance outputs to: {best_dest}")

print(f"Analysis plots saved to: {PLOTS_DIR}")
