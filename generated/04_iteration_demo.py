# -*- coding: utf-8 -*-
"""
手写 KMeans 迭代过程可视化（k-means++ 初始化）
标准化 income/balance → 手动循环 assign → update → 画 4 张快照
保存为 outputs/06_iteration.png
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import StandardScaler

# ---------- 中文显示设置 ----------
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans SC"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ---------- 读入 + 标准化 ----------
df = pd.read_csv("data/mall_customers_yuan.csv")
X_raw = df[["income", "balance"]]
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)   # 标准化后的 300×2 矩阵

# ========================================================
# 手写 k-means++ 初始化（和 sklearn 默认 init='k-means++' 一致）
# ========================================================
def kmeans_plus_plus_init(X, k, random_state=42):
    """k-means++：第一个随机，后续按距离平方概率选"""
    rng = np.random.RandomState(random_state)
    n, d = X.shape
    centers = np.empty((k, d))
    # 第 1 个：从 n 个样本里均匀抽一个
    idx = rng.randint(n)
    centers[0] = X[idx]
    # 后续 k-1 个：按 D(x)^2 概率抽样
    for i in range(1, k):
        # 每个点到最近已有质心的距离平方
        dists = np.min(np.linalg.norm(X[:, None, :] - centers[:i], axis=2), axis=1) ** 2
        probs = dists / dists.sum()
        idx = rng.choice(n, p=probs)
        centers[i] = X[idx]
    return centers


# ========================================================
# 手写 KMeans 主循环
# ========================================================
def run_kmeans_iter(X, k, random_state=42, max_iter=300, tol=1e-4):
    """
    返回所有轮次的快照：
      rounds[i] = {"centers": ..., "labels": ..., "inertia": ..., "shift": ...}
      rounds[0] 是初始质心 + 第一次分配的结果
      shift = 本轮质心 相对 上一轮质心 的最大位移（欧氏距离）
    """
    centers = kmeans_plus_plus_init(X, k, random_state)
    rounds = []
    prev_centers = None

    for i in range(max_iter):
        # assign：每个点分配到最近的质心
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)  # (n, k)
        labels = np.argmin(dists, axis=1)

        # inertia：簇内平方和
        inertia = np.sum(dists[np.arange(len(labels)), labels] ** 2)

        # shift：质心移动量（和上一轮比）
        if prev_centers is None:
            shift = None    # 第 0 轮没有上一轮
        else:
            shift = float(np.max(np.linalg.norm(centers - prev_centers, axis=1)))

        rounds.append({
            "iter": i,
            "centers": centers.copy(),
            "labels": labels.copy(),
            "inertia": inertia,
            "shift": shift,
        })

        # update：质心挪到簇内样本均值
        new_centers = np.empty_like(centers)
        for c in range(k):
            mask = labels == c
            if mask.sum() > 0:
                new_centers[c] = X[mask].mean(axis=0)
            else:
                new_centers[c] = X[np.random.RandomState(random_state).choice(len(X))]

        # 收敛判断：质心最大位移 < tol
        if prev_centers is not None and shift is not None and shift < tol:
            # 收敛了，记录最后一轮的 shift（很小）
            rounds[-1]["shift"] = shift if prev_centers is None else float(
                np.max(np.linalg.norm(new_centers - centers, axis=1))
            )
            centers = new_centers
            break

        prev_centers = centers.copy()
        centers = new_centers
    else:
        # max_iter 才出来
        rounds[-1]["shift"] = float(np.max(np.linalg.norm(new_centers - centers, axis=1)))

    return rounds


# ---------- 跑一遍 ----------
rounds = run_kmeans_iter(X, k=3, random_state=42)

print("=" * 65)
print("【手写 KMeans++ 迭代过程 · 每轮快照】")
print(f"{'轮次':>4s}  {'inertia':>10s}  {'质心位移(最大)':>14s}  {'质心坐标 (income_z, balance_z)':>40s}")
print("-" * 65)

for r in rounds:
    it = r["iter"]
    inert = r["inertia"]
    shift_str = f"{r['shift']:.6f}" if r["shift"] is not None else "— (初始质心)"
    centers_str = np.array2string(r["centers"].round(4), max_line_width=50)
    print(f"{it:>4d}  {inert:>10.4f}  {shift_str:>14s}  centers={centers_str}")

print(f"\n共迭代 {len(rounds)} 轮收敛。")
print("=" * 65)

# ========================================================
# 选 4 个快照来画：第 0 轮、第 1 轮、第 2 轮、最后一轮
# ========================================================
snapshot_indices = [0, 1, 2, len(rounds) - 1]
colors = ["#DD8452", "#4C72B0", "#55A868"]

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

for i, idx in enumerate(snapshot_indices):
    row, col = divmod(i, 2)
    ax = axes[row, col]
    r = rounds[idx]

    # 画散点：按当轮 labels 着色
    for c in range(3):
        mask = r["labels"] == c
        ax.scatter(
            X[mask, 0], X[mask, 1],
            s=40, alpha=0.7, color=colors[c],
            edgecolors="white", linewidths=0.5, zorder=2,
        )

    # 画质心：大星形
    ax.scatter(
        r["centers"][:, 0], r["centers"][:, 1],
        marker="*", s=600, c="red",
        edgecolors="black", linewidths=1.5, zorder=5,
    )

    # 画质心的编号，方便追踪每个质心的移动轨迹
    for ci, (cx, cy) in enumerate(r["centers"]):
        ax.annotate(
            f"C{ci}", (cx, cy),
            textcoords="offset points", xytext=(8, 8),
            fontsize=10, fontweight="bold", color="red",
        )

    # 标题：区分"第 N 轮"和"收敛"
    is_last = (idx == len(rounds) - 1)
    if is_last and r["shift"] is not None and r["shift"] < 1e-3:
        title = f"第 {idx} 轮：收敛（质心不再移动）\ninertia={r['inertia']:.2f}"
    elif r["shift"] is None:
        title = f"第 {idx} 轮：初始化（k-means++ 选初始质心）\ninertia={r['inertia']:.2f}"
    else:
        title = f"第 {idx} 轮：质心移动了 {r['shift']:.4f}\ninertia={r['inertia']:.2f}"

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("income（标准化 z-score）", fontsize=10)
    ax.set_ylabel("balance（标准化 z-score）", fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")

fig.suptitle("KMeans 迭代过程：分配-更新（k=3，k-means++ 初始化）", fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig("outputs/06_iteration.png", dpi=150, bbox_inches="tight")
print("\n已保存 outputs/06_iteration.png")
print("\n===== 迭代过程可视化完成 =====")
