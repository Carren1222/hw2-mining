# -*- coding: utf-8 -*-
"""
选 k：肘部法 + 轮廓系数
用标准化后的 income 和 balance，k 从 1 扫到 9，
同时打印每个 k 的数值表（终端），再画双图对比保存到 outputs/05_choose_k.png
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ---------- 中文显示设置 ----------
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans SC"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ---------- 读入数据 + 标准化 ----------
df = pd.read_csv("data/mall_customers_yuan.csv")
X_raw = df[["income", "balance"]]
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ========================================================
# 扫 k=1~9（肘部法需要 k 从 1 开始）
# ========================================================
ks = list(range(1, 10))

inertias = []        # 肘部法：簇内平方和
sil_scores = []     # 轮廓系数（k=1 时无意义，用 None 占位）

print("=" * 65)
print("【肘部法 + 轮廓系数 · k=1~9 扫描结果】")
print(f"{'k':>3s}  {'inertia_ (簇内平方和)':>22s}  {'silhouette_score':>18s}")
print("-" * 65)

for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)

    if k >= 2:
        sil = silhouette_score(X, labels)
        sil_scores.append(sil)
        print(f"{k:>3d}  {km.inertia_:>22.2f}  {sil:>18.4f}")
    else:
        sil_scores.append(np.nan)
        print(f"{k:>3d}  {km.inertia_:>22.2f}  {'— (k=1 无轮廓系数)':>18s}")

print("=" * 65)

# ========================================================
# 打印数值表（给用户看的正式表格）
# ========================================================
print("\n【肘部法 · 每个 k 的 inertia】")
elbow_table = pd.DataFrame({"k": ks, "inertia": inertias})
# 顺手算一下每个 k 比前一个 k 下降了多少，帮助肉眼找拐点
elbow_table["delta_inertia"] = elbow_table["inertia"].diff().round(2)
elbow_table["下降率"] = (elbow_table["delta_inertia"] / elbow_table["inertia"].shift(1) * 100).round(1)
elbow_table["下降率"].iloc[0] = np.nan
print(elbow_table.to_string(index=False))

print("\n【轮廓系数 · 每个 k 的 silhouette_score】")
sil_table = pd.DataFrame({"k": ks[1:], "silhouette": sil_scores[1:]})
sil_table["排名"] = sil_table["silhouette"].rank(ascending=False).astype(int)
sil_table = sil_table.sort_values("排名")
print(sil_table.to_string(index=False))

# ========================================================
# 双图对比可视化
# ========================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ---------- 左图：肘部法 ----------
ax = axes[0]
ax.plot(ks, inertias, marker="o", linewidth=2, markersize=8, color="#DD8452")
ax.set_xlabel("簇数 k", fontsize=12)
ax.set_ylabel("簇内平方和 inertia", fontsize=12)
ax.set_title("左图 · 肘部法：找 inertia 下降速度突然变缓的拐点", fontsize=13)
ax.set_xticks(ks)
ax.grid(alpha=0.3, linestyle="--")
# 标注每个点的数值
for k, v in zip(ks, inertias):
    ax.annotate(f"{v:.0f}", (k, v), textcoords="offset points",
                xytext=(0, -15), ha="center", fontsize=8, color="#666666")

# ---------- 右图：轮廓系数 ----------
ax = axes[1]
valid_k = ks[1:]          # k=2~9
valid_sil = sil_scores[1:]
# 用最高的那个 k 画个虚线参考线
best_idx = int(np.argmax(valid_sil))
best_k = valid_k[best_idx]
best_sil = valid_sil[best_idx]

ax.plot(valid_k, valid_sil, marker="s", linewidth=2, markersize=8, color="#4C72B0")
ax.axvline(x=best_k, color="red", linestyle="--", alpha=0.6,
           label=f"最高点 k={best_k}，sil={best_sil:.3f}")
ax.scatter([best_k], [best_sil], marker="*", s=300, color="red", zorder=5)
ax.set_xlabel("簇数 k", fontsize=12)
ax.set_ylabel("轮廓系数 silhouette_score", fontsize=12)
ax.set_title("右图 · 轮廓系数：数值越大，簇内越紧、簇间越远", fontsize=13)
ax.set_xticks(valid_k)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, linestyle="--")
# 标注每个点的数值
for k, v in zip(valid_k, valid_sil):
    ax.annotate(f"{v:.3f}", (k, v), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=8, color="#666666")

fig.suptitle("选 k：两种指标对比（数据已标准化，KMeans random_state=42, n_init=10）", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("outputs/05_choose_k.png", dpi=150, bbox_inches="tight")
print("\n已保存 outputs/05_choose_k.png")
print("\n===== 选 k 扫描完成 =====")
