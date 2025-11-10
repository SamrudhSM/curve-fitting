import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


df = pd.read_csv("xy_data.csv")
x_data = df["x"].values.astype(float)
y_data = df["y"].values.astype(float)
N = len(df)


t_data = np.linspace(6.0, 60.0, N, endpoint=True)


def model_xy(t, theta_deg, M, X):
    theta = np.deg2rad(theta_deg)
    exp_term = np.exp(M * np.abs(t))
    sin03 = np.sin(0.3 * t)
    x = t * np.cos(theta) - exp_term * sin03 * np.sin(theta) + X
    y = 42.0 + t * np.sin(theta) + exp_term * sin03 * np.cos(theta)
    return x, y

def l1_loss(params):
    theta_deg, M, X = params
    penalty = 0.0
    if not (0.0 < theta_deg < 50.0):
        d = max(0.0, 0.0 - theta_deg, theta_deg - 50.0)
        penalty += 1e6 * (1.0 + d**2)
    if not (-0.05 < M < 0.05):
        d = max(0.0, -0.05 - M, M - 0.05)
        penalty += 1e6 * (1.0 + 20000.0 * d**2)
    if not (0.0 < X < 100.0):
        d = max(0.0, 0.0 - X, X - 100.0)
        penalty += 1e6 * (1.0 + 0.01 * d**2)

    x_pred, y_pred = model_xy(t_data, theta_deg, M, X)
    loss = np.sum(np.abs(x_data - x_pred) + np.abs(y_data - y_pred))
    return loss + penalty

def clip_params(p):
    p = np.array(p, dtype=float)
    p[0] = np.clip(p[0], 0.0, 50.0)     
    p[1] = np.clip(p[1], -0.05, 0.05)   
    p[2] = np.clip(p[2], 0.0, 100.0)    
    return p

thetas = np.linspace(0.0, 50.0, 51)    
Ms     = np.linspace(-0.05, 0.05, 11)  
Xs     = np.linspace(0.0, 100.0, 51)   

best = None
best_loss = np.inf
for th in thetas:
    th_rad = np.deg2rad(th)
    cos_th, sin_th = np.cos(th_rad), np.sin(th_rad)
    for Mm in Ms:
        exp_term = np.exp(Mm * np.abs(t_data))
        sin03 = np.sin(0.3 * t_data)
        base_x = t_data * cos_th - exp_term * sin03 * sin_th
        base_y = 42.0 + t_data * sin_th + exp_term * sin03 * cos_th
        for Xc in Xs:
            loss = np.sum(np.abs(x_data - (base_x + Xc)) + np.abs(y_data - base_y))
            if loss < best_loss:
                best_loss = loss
                best = [th, Mm, Xc]

seed = np.array(best, dtype=float)


rng = np.random.default_rng(123)
rand_best = seed.copy()
rand_best_loss = l1_loss(rand_best)
scales = np.array([2.0, 0.005, 2.0])

for _ in range(2000):
    cand = clip_params(rand_best + rng.normal(0.0, scales))
    loss = l1_loss(cand)
    if loss < rand_best_loss:
        rand_best = cand
        rand_best_loss = loss

res = minimize(l1_loss, rand_best, method="Nelder-Mead",
               options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-3})
theta_hat, M_hat, X_hat = res.x.tolist()
final_loss = l1_loss(res.x)


x_pred, y_pred = model_xy(t_data, theta_hat, M_hat, X_hat)

plt.figure()
plt.scatter(x_data, y_data, s=8, label="Data")
plt.plot(x_pred, y_pred, linewidth=2, label="Predicted")
plt.title("Data vs Predicted Curve")
plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.grid(True)
plt.savefig("data_vs_predicted.png", bbox_inches="tight")

err = np.abs(x_data - x_pred) + np.abs(y_data - y_pred)
plt.figure()
plt.plot(t_data, err, linewidth=2)
plt.title("Per-point L1 Error vs t")
plt.xlabel("t"); plt.ylabel("|Δx| + |Δy|"); plt.grid(True)
plt.savefig("error_vs_t.png", bbox_inches="tight")

results = {
    "theta_deg": float(theta_hat),
    "M": float(M_hat),
    "X": float(X_hat),
    "final_L1_loss": float(final_loss),
    "seed_params_from_grid": [float(v) for v in seed],
    "grid_best_loss": float(best_loss),
    "rand_best_params": [float(v) for v in rand_best],
    "rand_best_loss": float(rand_best_loss),
    "scipy_success": bool(res.success),
    "scipy_message": str(res.message),
}
with open("fit_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
print("\nSaved: data_vs_predicted.png, error_vs_t.png, fit_results.json")
