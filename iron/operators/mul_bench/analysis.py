import pandas as pd
import numpy as np
import os
from scipy.stats import linregress

summary_path = '/home/steven/gpu_bench/IRON/iron/operators/mul_bench/counter_verify/out_sweep/flops_sweep_summary.csv'
df_summary = pd.read_csv(summary_path)

rs = []
actual_flops = []
sum_sq_insts = []

for index, row in df_summary.iterrows():
    r = row['R']
    flops = row['actual_flops_total']
    csv_path = row['csv_path']
    
    if os.path.exists(csv_path):
        df_csv = pd.read_csv(csv_path)
        # Skip empty lines if any
        df_csv = df_csv.dropna(subset=['SQ_INSTS_VALU'])
        sq_sum = df_csv['SQ_INSTS_VALU'].sum()
        
        rs.append(r)
        actual_flops.append(flops)
        sum_sq_insts.append(sq_sum)
    else:
        print(f"Warning: {csv_path} not found")

rs = np.array(rs)
actual_flops = np.array(actual_flops)
sum_sq_insts = np.array(sum_sq_insts)

# Linear fit: actual_flops vs R
slope1, intercept1, r_value1, p_value1, std_err1 = linregress(rs, actual_flops)
r_squared1 = r_value1**2

# Linear fit: sum_sq_insts vs R
slope2, intercept2, r_value2, p_value2, std_err2 = linregress(rs, sum_sq_insts)
r_squared2 = r_value2**2

# Baseline-subtracted
min_r_idx = np.argmin(rs)
min_r = rs[min_r_idx]
min_flops = actual_flops[min_r_idx]
deltas = actual_flops - min_flops

print("Analysis Results:")
print(f"Fit 1 (actual_flops_total vs R):")
print(f"  Slope: {slope1}")
print(f"  Intercept: {intercept1}")
print(f"  R^2: {r_squared1}")
print(f"Fit 2 (sum(SQ_INSTS_VALU) vs R):")
print(f"  Slope: {slope2}")
print(f"  Intercept: {intercept2}")
print(f"  R^2: {r_squared2}")
print("\nFirst 8 rows (R, delta):")
for i in range(min(8, len(rs))):
    print(f"  R={rs[i]}, Delta={deltas[i]}")
