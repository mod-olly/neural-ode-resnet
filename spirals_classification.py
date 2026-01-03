#!/usr/bin/env python3
"""
spirals_classification.py
Обучение на двухклассовых 2D-спиралях: сравнение ResNet (stacked residual blocks),
Neural ODE (fixed-step via odeint with fixed t grid) и Neural ODE (adaptive solver).
Сохраняет результаты и печатает accuracy / runtime.
"""
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_spiral   # sklearn might not have make_spiral: we'll implement if absent
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# try to import torchdiffeq
try:
    from torchdiffeq import odeint
except Exception as e:
    raise RuntimeError("torchdiffeq is required. Install with: pip install torchdiffeq") from e

# reproducibility
SEED = 123
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# generate two-class 2D spiral dataset
def make_two_spirals(n_points=1000, noise=0.2):
    # custom implementation (if sklearn lacks make_spiral)
    n = n_points // 2
    theta = np.sqrt(np.random.rand(n)) * 2 * np.pi  # np.linspace(0,2pi,n)
    r_a = 2*theta + np.pi
    data_a = np.vstack([np.cos(theta)*r_a, np.sin(theta)*r_a]).T + noise*np.random.randn(n,2)

    theta = np.sqrt(np.random.rand(n)) * 2 * np.pi
    r_b = -2*theta - np.pi
    data_b = np.vstack([np.cos(theta)*r_b, np.sin(theta)*r_b]).T + noise*np.random.randn(n,2)

    X = np.vstack([data_a, data_b])
    y = np.hstack([np.zeros(n), np.ones(n)])
    return X, y

# small MLP used as vector field f
class ODEFunc(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 2)
        )
        # initialize small
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, t, x):
        # odeint passes t first
        return self.net(x)

# ResNet residual block (single-block increments)
class ResidualBlock(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
    def forward(self, x):
        return x + self.block(x)

# Simple classifier: embed 2D -> hidden -> apply several residual/Euler steps -> classifier head
class ResNetClassifier(nn.Module):
    def __init__(self, n_blocks=4, hidden_dim=32):
        super().__init__()
        self.embed = nn.Linear(2, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden_dim, 2)
    def forward(self, x):
        h = torch.tanh(self.embed(x))
        for b in self.blocks:
            h = b(h)
        return self.head(h)

# Neural ODE classifier (use ODEFunc as continuous block)
class NeuralODEClassifier(nn.Module):
    def __init__(self, ode_func: ODEFunc, t_span, solver='rk4', hidden_dim=32):
        super().__init__()
        self.embed = nn.Linear(2, hidden_dim)
        self.odefunc = ode_func
        self.head = nn.Linear(hidden_dim, 2)
        self.t_span = t_span  # tensor of times to evaluate (for fixed-step)
        self.solver = solver
    def forward(self, x):
        h0 = torch.tanh(self.embed(x))
        # odeint: input shape (batch, dim) --> odeint expects (batch, dim) but some versions need (dim,)
        # torchdiffeq's odeint works on tensors of shape (batch, dim)
        # we integrate along time points t_span and take the final state
        # if t_span has length >2 this is fixed-step evaluation (many evals)
        # for adaptive solver we pass a short t_span = [0,1] and solver='dopri5'
        hT = odeint(self.odefunc, h0, self.t_span.to(h0.device), method=self.solver)
        # odeint returns shape (len(t_span), batch, dim)
        h_final = hT[-1]
        return self.head(h_final)

# training utilities
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device).float(); yb = yb.to(device).long()
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device).float(); yb = yb.to(device).long()
            logits = model(xb)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
    return correct / total

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    outdir = "results_spirals"
    os.makedirs(outdir, exist_ok=True)

    # dataset
    X, y = make_two_spirals(n_points=2000, noise=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)
    tr_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    te_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True)
    te_loader = DataLoader(te_ds, batch_size=256, shuffle=False)

    # common training config
    epochs = 60
    lr = 1e-3

    experiments = []

    # 1) ResNet variants with different depths (simulate different effective h by more residual blocks)
    for n_blocks in [2, 4, 8]:
        model = ResNetClassifier(n_blocks=n_blocks, hidden_dim=32).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        t0 = time.time()
        for ep in range(epochs):
            train_epoch(model, tr_loader, criterion, optimizer)
        train_time = time.time() - t0
        acc = evaluate(model, te_loader)
        experiments.append({
            'name': f'ResNet_blocks_{n_blocks}',
            'params': count_parameters(model),
            'time': train_time,
            'acc': acc
        })
        print("ResNet", n_blocks, "acc", acc, "time", train_time)

    # 2) Neural ODE fixed-step: we simulate with a t_span of many time points (so many function evals)
    for n_steps in [2, 4, 8, 16]:  # number of eval points (fixed grid)
        t_span = torch.linspace(0., 1., n_steps+1)  # +1 because includes t0 and tN
        ode_func = ODEFunc(dim=32).to(device)
        model = NeuralODEClassifier(ode_func, t_span=t_span, solver='rk4', hidden_dim=32).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        t0 = time.time()
        for ep in range(epochs):
            train_epoch(model, tr_loader, criterion, optimizer)
        train_time = time.time() - t0
        acc = evaluate(model, te_loader)
        experiments.append({
            'name': f'NODE_fixed_{n_steps}',
            'params': count_parameters(model),
            'time': train_time,
            'acc': acc,
            'n_steps': n_steps
        })
        print("NODE fixed steps", n_steps, "acc", acc, "time", train_time)

    # 3) Neural ODE adaptive solver (dopri5)
    t_span_short = torch.tensor([0., 1.])  # adaptive solver will take internal steps
    for tol in [1e-3, 1e-4, 1e-5]:
        ode_func = ODEFunc(dim=32).to(device)
        model = NeuralODEClassifier(ode_func, t_span=t_span_short, solver='dopri5', hidden_dim=32).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        t0 = time.time()
        for ep in range(epochs):
            train_epoch(model, tr_loader, criterion, optimizer)
        train_time = time.time() - t0
        acc = evaluate(model, te_loader)
        experiments.append({
            'name': f'NODE_adaptive_tol_{tol}',
            'params': count_parameters(model),
            'time': train_time,
            'acc': acc,
            'tol': tol
        })
        print("NODE adaptive tol", tol, "acc", acc, "time", train_time)

    # save results
    import json
    with open(os.path.join(outdir, "experiments.json"), "w") as f:
        json.dump(experiments, f, indent=2)
    print("Saved experiments to", outdir)

if __name__ == "__main__":
    main()
