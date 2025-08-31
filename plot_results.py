## NGL, this is AI generated, got better things to do with my time fr fr

import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("results.csv")

# --- Convergence Plot ---
plt.figure(figsize=(8,5))
plt.plot(df["nPaths"], df["GPUPrice"], "o-", label="GPU Monte Carlo")
plt.plot(df["nPaths"], df["CPUPrice"], "s-", label="CPU Monte Carlo")
plt.axhline(y=df["BlackScholes"][0], color="red", linestyle="--", label="Black-Scholes")
plt.xscale("log")
plt.xlabel("Number of Paths (log scale)")
plt.ylabel("Option Price")
plt.title("Monte Carlo Convergence to Black-Scholes")
plt.legend()
plt.grid(True)
plt.savefig("convergence.png")

# --- Runtime Comparison ---
plt.figure(figsize=(8,5))
plt.plot(df["nPaths"], df["CPUTime"], "s-", label="CPU Time")
plt.plot(df["nPaths"], df["GPUTime"], "o-", label="GPU Time")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Number of Paths (log scale)")
plt.ylabel("Runtime (seconds, log scale)")
plt.title("CPU vs GPU Runtime")
plt.legend()
plt.grid(True, which="both")
plt.savefig("runtime.png")

print("Plots saved: convergence.png, runtime.png")
