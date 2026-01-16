import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

data = pd.read_csv("simulation_results/finger_boundary_stress_all.csv")

a_values = []
b_values = []

for i in range(len(data)):
    x_list = np.linspace(0,1000,10000)
    x = [1,10,20,30,40,50,60,70,80,90,100, 1000,10000]
    y_before = data.iloc[i].tolist()
    y = y_before[2:]

    slope, intercept, _, _,_ = linregress(x,y)
    a_values.append(intercept)
    b_values.append(slope)

a_values = np.array(a_values)
b_values = np.array(b_values)

print(f"Mean a value: {np.mean(a_values)}")
print(f"Mean b value: {np.mean(b_values)}")

print(f"SD a value: {np.std(a_values)}")
print(f"SD b value: {np.std(b_values)}")

print(f"Max a value: {np.max(a_values)}")
print(f"Max b value: {np.max(b_values)}")

print(f"Min a value: {np.min(a_values)}")
print(f"Min b value: {np.min(b_values)}")