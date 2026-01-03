#!/usr/bin/env python3
"""
toy_dynamics.py
Игрушечная 2D динамика: сравнение ResNet (Euler) vs NODE (RK4 fixed) vs Adaptive RK4.
Сохраняет графики и CSV в ./results/
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import perf_counter

# reproducible
SEED = 1
np.random.seed(SEED)

def make_f_theta(seed=0):
    rng = np.random.RandomState(seed)
    W1 = rng.randn(8, 2) * 0.8
    b1 = rng.randn(8) * 0.1
    W2 = rng.randn(2, 8) * 0.8
    def f(x):
        # x can be (2,) or (n,2)
        x2 = np.atleast_2d(x)
        h = np.tanh(x2.dot(W1.T) + b1)
        out = h.dot(W2.T)
        return out if x.ndim==2 else out[0]
    return f

# RK4 single step
def rk4_step(f, x, h):
    k1 = f(x)
    k2 = f(x + 0.5*h*k1)
    k3 = f(x + 0.5*h*k2)
    k4 = f(x + h*k3)
    return x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# integrate with RK4 fixed-step
def integrate_rk4(f, x0, t0, t1, dt):
    steps = int(np.ceil((t1 - t0)/dt))
    xs = np.zeros((steps+1, 2))
    ts = np.zeros(steps+1)
    xs[0] = x0
    ts[0] = t0
    x = x0.copy()
    t = t0
    for i in range(1, steps+1):
        h = min(dt, t1 - t)
        x = rk4_step(f, x, h)
        t += h
        xs[i] = x
        ts[i] = t
    return ts, xs

# simple adaptive RK4 using step-doubling
def integrate_adaptive(f, x0, t0, t1, h0=0.1, tol=1e-6, h_min=1e-6, h_max=0.5):
    t = t0
    x = x0.copy()
    ts = [t]
    xs = [x.copy()]
    h = h0
    while t < t1 - 1e-12:
        h = min(h, t1 - t)
        # one full step
        x_full = rk4_step(f, x, h)
        # two half steps
        x_half = rk4_step(f, x, h*0.5)
        x_half = rk4_step(f, x_half, h*0.5)
        err = np.linalg.norm(x_full - x_half) / max(1.0, np.linalg.norm(x_half))
        if err <= tol:
            t += h
            x = x_half
            ts.append(t); xs.append(x.copy())
            # adapt h
            if err < 1e-12:
                s = 2.0
            else:
                s = min(2.0, 0.9*(tol/err)**0.25)
            h = min(h_max, h*s)
        else:
            s = max(0.1, 0.9*(tol/err)**0.25)
            h = max(h_min, h*s)
    return np.array(ts), np.array(xs)

# Euler integrate (ResNet)
def integrate_euler(f, x0, t0, t1, N):
    h = (t1 - t0)/N
    x = x0.copy()
    xs = np.zeros((N+1, 2)); ts = np.zeros(N+1)
    xs[0] = x.copy(); ts[0] = t0
    t = t0
    for k in range(1, N+1):
        x = x + h * f(x)
        t += h
        xs[k] = x.copy(); ts[k] = t
    return ts, xs

# RK2 midpoint (two-stage residual)
def integrate_rk2(f, x0, t0, t1, N):
    h = (t1 - t0)/N
    x = x0.copy()
    xs = np.zeros((N+1, 2)); ts = np.zeros(N+1)
    xs[0] = x.copy(); ts[0] = t0
    t = t0
    for k in range(1, N+1):
        k1 = f(x)
        xm = x + 0.5*h*k1
        k2 = f(xm)
        x = x + h*k2
        t += h
        xs[k] = x.copy(); ts[k] = t
    return ts, xs

def sample_reference(ts_query, t_ref, x_ref):
    # nearest-sample interpolation for reference
    dt_ref = t_ref[1]-t_ref[0]
    xs_out = []
    t0 = t_ref[0]
    for tq in ts_query:
        idx = int(round((tq - t0)/dt_ref))
        idx = min(max(0, idx), len(x_ref)-1)
        xs_out.append(x_ref[idx])
    return np.array(xs_out)

def main():
    outdir = "results_toy"
    os.makedirs(outdir, exist_ok=True)

    f = make_f_theta(seed=SEED)
    x0 = np.array([1.0, 0.0])
    t0 = 0.0; t1 = 2.0

    # reference: RK4 with tiny dt
    dt_ref = 1e-4
    t_ref, x_ref = integrate_rk4(f, x0, t0, t1, dt_ref)

    hs = [0.5, 0.25, 0.125, 0.0625, 0.03125]
    records = []
    for h in hs:
        N = int(np.round((t1 - t0)/h))
        h_actual = (t1 - t0)/N

        t0t = perf_counter(); ts_e, xs_e = integrate_euler(f, x0, t0, t1, N); te = perf_counter()-t0t
        t0t = perf_counter(); ts_rk4, xs_rk4 = integrate_rk4(f, x0, t0, t1, h_actual); trk4 = perf_counter()-t0t
        t0t = perf_counter(); ts_ad, xs_ad = integrate_adaptive(f, x0, t0, t1, h0=h_actual, tol=1e-6); tad = perf_counter()-t0t

        xref_e = sample_reference(ts_e, t_ref, x_ref)
        xref_rk4 = sample_reference(ts_rk4, t_ref, x_ref)
        xref_ad = sample_reference(ts_ad, t_ref, x_ref)

        err_e = np.mean(np.linalg.norm(xs_e - xref_e, axis=1))
        err_r = np.mean(np.linalg.norm(xs_rk4 - xref_rk4, axis=1))
        err_a = np.mean(np.linalg.norm(xs_ad - xref_ad, axis=1))

        records.append({
            'h': h_actual, 'N': N,
            'err_euler': err_e, 'time_euler': te,
            'err_rk4': err_r, 'time_rk4': trk4,
            'err_adaptive': err_a, 'time_adaptive': tad,
            'steps_adaptive': len(ts_ad)-1
        })

    df = pd.DataFrame.from_records(records)
    df.to_csv(os.path.join(outdir, "trajectory_comparison.csv"), index=False)
    print("Saved CSV to", outdir)

    # Plots
    plt.figure(figsize=(6,5))
    plt.loglog(df['h'], df['err_euler'], 'o-', label='Euler (ResNet)')
    plt.loglog(df['h'], df['err_rk4'], 's-', label='RK4 fixed-step (NODE)')
    plt.loglog(df['h'], df['err_adaptive'], 'd-', label='Adaptive RK4')
    plt.xlabel('h'); plt.ylabel('mean trajectory error'); plt.title('Error vs h')
    plt.legend(); plt.grid(True, which='both', ls='--', lw=0.5)
    plt.savefig(os.path.join(outdir, "error_vs_h.png"), dpi=150); plt.close()

    # Phase example
    h_ex = 0.125; N_ex = int(round((t1 - t0)/h_ex))
    _, xs_e_ex = integrate_euler(f, x0, t0, t1, N_ex)
    _, xs_rk_ex = integrate_rk4(f, x0, t0, t1, h_ex)
    ts_ad_ex, xs_ad_ex = integrate_adaptive(f, x0, t0, t1, h0=h_ex, tol=1e-6)

    plt.figure(figsize=(6,5))
    plt.plot(x_ref[:,0], x_ref[:,1], '-', label='reference (RK4 dt=1e-4)')
    plt.plot(xs_e_ex[:,0], xs_e_ex[:,1], 'o-', label=f'Euler h={h_ex}')
    plt.plot(xs_rk_ex[:,0], xs_rk_ex[:,1], 's-', label=f'RK4 fixed h={h_ex}')
    plt.plot(xs_ad_ex[:,0], xs_ad_ex[:,1], 'd-', label='Adaptive RK4')
    plt.legend(); plt.xlabel('x1'); plt.ylabel('x2'); plt.title('Phase trajectories (example)')
    plt.grid(True); plt.savefig(os.path.join(outdir, "phase_example.png"), dpi=150); plt.close()

    # Convergence order plot
    Ns = np.array([10, 20, 40, 80, 160, 320])
    errs = []
    for N in Ns:
        _, xs_e = integrate_euler(f, x0, t0, t1, N)
        _, xs_m = integrate_rk2(f, x0, t0, t1, N)
        hN = (t1 - t0)/N
        _, xs_r = integrate_rk4(f, x0, t0, t1, hN)
        err_e = np.mean(np.linalg.norm(xs_e - sample_reference(np.linspace(t0,t1,N+1), t_ref,x_ref), axis=1))
        err_m = np.mean(np.linalg.norm(xs_m - sample_reference(np.linspace(t0,t1,N+1), t_ref,x_ref), axis=1))
        err_r = np.mean(np.linalg.norm(xs_r - sample_reference(np.linspace(t0,t1,N+1), t_ref,x_ref), axis=1))
        errs.append((err_e, err_m, err_r))
    errs = np.array(errs)

    plt.figure(figsize=(6,5))
    plt.loglog(Ns, errs[:,0], 'o-', label='Euler (~order 1)')
    plt.loglog(Ns, errs[:,1], 's-', label='RK2 (~order 2)')
    plt.loglog(Ns, errs[:,2], 'd-', label='RK4 (~order 4)')
    plt.xlabel('N'); plt.ylabel('mean trajectory error'); plt.title('Convergence order')
    plt.legend(); plt.grid(True, which='both', ls='--', lw=0.5)
    plt.savefig(os.path.join(outdir, "convergence_order.png"), dpi=150); plt.close()

    print(df)
    print("Plots and csv saved into", outdir)

if __name__ == "__main__":
    main()
