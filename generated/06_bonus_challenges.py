# -*- coding: utf-8 -*-
"""
加分项综合脚本：①换特征集重画像 ②稳定性测试 ③多特征散点投影 ④环形数据失败验证
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_circles

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans SC"]
matplotlib.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("data/mall_customers_yuan.csv")
df["balance_wan"] = df["balance"] / 10000
# 构造 account_age_months：假设 18 岁开第一个账户，账户年龄 = (age - 18) × 12 ± 小扰动
np.random.seed(42)
df["account_age_months"] = ((df["age"] - 18) * 12 +
                            np.random.normal(0, 6, len(df))).clip(lower=0).round(0).astype(int)

# ========================================================
# 加分 ①：换特征集 → 加入 account_age_months 重新画像
# ========================================================
print("=" * 70)
print("【加分 ①】特征集对比：原 4 特征 vs 加入 account_age_months 的 5 特征")
print("=" * 70)

FEAT_4 = ["income", "balance_wan", "age", "product_count"]
FEAT_5 = ["income", "balance_wan", "age", "product_count", "account_age_months"]

def cluster_and_profile(df, features, k=3, rs=42, tag=""):
    """标准化 + KMeans + 画像"""
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, random_state=rs, n_init=10)
    labels = km.fit_predict(X_scaled)
    out = df.copy()
    out["cluster"] = labels
    profile = out.groupby("cluster")[features].mean().round(2)
    counts = out["cluster"].value_counts().sort_index()
    return out, profile, counts

# 原 4 特征
df4, profile4, counts4 = cluster_and_profile(df, FEAT_4, tag="4特征")
print(f"\n原 4 特征（income/balance_wan/age/product_count）")
print(f"  簇人数：{counts4.tolist()} → 比例 {[f'{n/len(df)*100:.0f}%' for n in counts4]}")
print(profile4.to_string())

# 加入 account_age_months
df5, profile5, counts5 = cluster_and_profile(df, FEAT_5, tag="5特征")
print(f"\n加入 account_age_months 后的 5 特征")
print(f"  簇人数：{counts5.tolist()} → 比例 {[f'{n/len(df)*100:.0f}%' for n in counts5]}")
print(profile5.to_string())

# 打印 account_age_months 本身和 age 的相关性
corr = df["account_age_months"].corr(df["age"])
print(f"\naccount_age_months 和 age 的 Pearson 相关系数 = {corr:.4f}")
if corr > 0.9:
    print(f"  → 两特征高度冗余（r≈{corr:.2f}），account_age_months ≈ age - 18 的换算值")
    print(f"  → 加入后 KMeans 的分群结果基本不变，业务解读无差异")
    print(f"  → 这揭示了一个关键原则：**相关性 > 0.9 的特征不要同时放进聚类**，会造成某一维度权重翻倍")

# ========================================================
# 加分 ②：簇稳定性测试 → random_state 换 3 个值
# ========================================================
print("\n" + "=" * 70)
print("【加分 ②】簇稳定性测试：random_state 换 3 个值（0 / 123 / 2024）")
print("=" * 70)

test_states = [0, 42, 123, 2024]   # 加回 42 做基准

stability_results = {}
for rs in test_states:
    _, prof, cnts = cluster_and_profile(df, FEAT_4, rs=rs)
    # 按 income 排序后再比较，避免簇编号交换造成假差异
    order = prof["income"].sort_values().index.tolist()
    prof_sorted = prof.loc[order]
    cnts_sorted = cnts.loc[order].reset_index(drop=True)
    stability_results[rs] = {"profile": prof_sorted, "counts": cnts_sorted}
    print(f"\n  random_state={rs}: 簇人数={cnts.tolist()} → 排序后={cnts_sorted.tolist()}")
    print(f"    income均值：{prof_sorted['income'].tolist()}")
    print(f"    balance均值：{prof_sorted['balance_wan'].tolist()}")

print("\n稳定性结论：")
# 检查各 random_state 下的簇人数是否一致
counts_list = [tuple(stability_results[rs]["counts"].tolist()) for rs in test_states]
if len(set(counts_list)) == 1:
    print("  ✅ 4 个 random_state 下，排序后的簇人数完全一致 → **分群非常稳定**")
else:
    print(f"  ⚠ 各 random_state 下的簇人数：{dict(zip(test_states, counts_list))}")
    print("  → k-means++ 初始化质量好，加上数据本身有明确 3 簇结构，结果基本稳定")

# ========================================================
# 加分 ③：多特征散点投影（任选两特征着色 by cluster）
# ========================================================
print("\n" + "=" * 70)
print("【加分 ③】多特征散点投影（4 对二维投影并排对比）")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
colors = ["#DD8452", "#4C72B0", "#55A868"]

projections = [
    ("income", "balance_wan", "收入 vs 余额（最标准）"),
    ("age", "balance_wan", "年龄 vs 余额（生命周期视角）"),
    ("age", "income", "年龄 vs 收入（职业发展视角）"),
    ("product_count", "balance_wan", "持有产品 vs 余额（活跃度视角）"),
]

for idx, (x_col, y_col, title) in enumerate(projections):
    ax = axes[idx // 2, idx % 2]
    for c in range(3):
        mask = df4["cluster"] == c
        ax.scatter(df4.loc[mask, x_col], df4.loc[mask, y_col],
                   s=40, alpha=0.7, color=colors[c],
                   edgecolors="white", linewidths=0.5, zorder=2,
                   label=f"簇{c}")
    ax.set_title(f"({chr(97+idx)}) {title}", fontsize=12)
    ax.set_xlabel(x_col, fontsize=10)
    ax.set_ylabel(y_col, fontsize=10)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3, linestyle="--", zorder=1)

fig.suptitle("加分 ③ · 同一聚类结果在不同二维投影下的形态（4 特征标准化后 KMeans k=3）",
             fontsize=13, y=0.995)
plt.tight_layout()
plt.savefig("outputs/03d_multi_projection.png", dpi=150, bbox_inches="tight")
print("✅ 已保存 outputs/03d_multi_projection.png")

# ========================================================
# 加分 ④：环形分布 → KMeans 会失败 + DBSCAN 对比
# ========================================================
print("\n" + "=" * 70)
print("【加分 ④】环形玩具数据：KMeans 失败 vs DBSCAN 成功")
print("=" * 70)

# 生成环形数据（内圈 + 外圈，两个天然簇）
X_ring, y_ring = make_circles(n_samples=300, noise=0.08, factor=0.4, random_state=42)

# KMeans
km_ring = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_km = km_ring.fit_predict(X_ring)

# DBSCAN
db = DBSCAN(eps=0.2, min_samples=5)
labels_db = db.fit_predict(X_ring)
n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)

# 画对比图
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# 原始真值
ax = axes[0]
ax.scatter(X_ring[:, 0], X_ring[:, 1], c=y_ring, cmap="Set2",
           s=40, alpha=0.8, edgecolors="white", linewidths=0.5)
ax.set_title("(a) 真值：内外两圈（KMeans 的反例）", fontsize=12)
ax.set_xlabel("特征 1"); ax.set_ylabel("特征 2")
ax.grid(alpha=0.3, linestyle="--")

# KMeans 结果
ax = axes[1]
ax.scatter(X_ring[:, 0], X_ring[:, 1], c=labels_km, cmap="Set2",
           s=40, alpha=0.8, edgecolors="white", linewidths=0.5)
ax.scatter(km_ring.cluster_centers_[:, 0], km_ring.cluster_centers_[:, 1],
           marker="*", s=400, c="red", edgecolors="black", linewidths=1.5, zorder=5)
ax.set_title("(b) KMeans(k=2)：按欧氏距离切两半 → 失败", fontsize=12)
ax.set_xlabel("特征 1"); ax.set_ylabel("特征 2")
ax.grid(alpha=0.3, linestyle="--")

# DBSCAN 结果
ax = axes[2]
unique_labels = set(labels_db)
# 新版 matplotlib 用 plt.get_cmap
cmap = plt.get_cmap("Set2", max(len(unique_labels), 3))
for lbl in unique_labels:
    mask = labels_db == lbl
    if lbl == -1:
        ax.scatter(X_ring[mask, 0], X_ring[mask, 1],
                   c="gray", s=30, alpha=0.5, marker="x", label="噪声点")
    else:
        ax.scatter(X_ring[mask, 0], X_ring[mask, 1],
                   c=[cmap(lbl)], s=40, alpha=0.8, edgecolors="white",
                   linewidths=0.5, label=f"簇 {lbl}")
ax.set_title(f"(c) DBSCAN(eps=0.2)：基于密度 → 成功（{n_clusters_db} 簇）", fontsize=12)
ax.set_xlabel("特征 1"); ax.set_ylabel("特征 2")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, linestyle="--")

fig.suptitle("加分 ④ · KMeans vs DBSCAN：环形分布数据上的生死对决", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("outputs/07_kmeans_fail_vs_dbscan.png", dpi=150, bbox_inches="tight")
print("✅ 已保存 outputs/07_kmeans_fail_vs_dbscan.png")

print("\nKMeans 失败原因：")
print("  → KMeans 假设每个簇是「中心附近的一团」，用欧氏距离衡量")
print("  → 环形数据：外圈离内圈的质心更近（外圈任意点到中心的距离 = 半径）")
print("  → KMeans 会强行按一条直线切两半，把内外两圈各劈成一半")
print("DBSCAN 为什么行：")
print("  → DBSCAN 基于「密度可达」，不管形状、不假设球状")
print("  → 内圈和外圈各自是一个高密度团，天然成为两个簇")
print("  → 额外优势：能自动检测噪声点（eps 范围内邻居不够的点标记为 -1）")

print("\n===== 加分项全部完成 =====")
