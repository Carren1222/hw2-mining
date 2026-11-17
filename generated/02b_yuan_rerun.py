# -*- coding: utf-8 -*-
"""
基于 mall_customers_yuan.csv 的 KMeans 聚类重跑（原封不动版）
balance 列为元（而非万元），income 为万元，两特征量纲不同。
本脚本严格按照"原数据、原参数、原流程"重跑，不加任何标准化或单位换算。
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans

# ---------- 中文显示设置 ----------
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans SC"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ========================================================
# 第 1 步：数据初检
# ========================================================
df = pd.read_csv("data/mall_customers_yuan.csv")

print("=" * 60)
print("【第 1 步 · 数据初检】")
print(f"行列数：{df.shape[0]} 行 × {df.shape[1]} 列")
print("\n各列缺失值：")
print(df.isna().sum())
print("\n数值列基本统计（均值、最小值、最大值）：")
num_cols = ["age", "income", "balance", "product_count"]
stats = df[num_cols].agg(["mean", "min", "max"]).round(2)
print(stats.to_string())

# 关键：balance 的量级提醒
print(f"\n⚠ 观察：income 均值 ≈ {df['income'].mean():.2f}（万元），"
      f"balance 均值 ≈ {df['balance'].mean():.2f}（元）——"
      f"两特征量纲差 {df['balance'].mean() / df['income'].mean():.0f} 倍。")
print("=" * 60)

# ========================================================
# 第 2 步：KMeans 聚类（原封不动，不改单位、不加标准化）
# ========================================================
X = df[["income", "balance"]]           # 只用 income 和 balance 两列
km = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = km.fit_predict(X)
df["cluster"] = labels

print("\n【第 2 步 · KMeans 聚类完成】")
print(f"簇中心坐标（income 万元, balance 元）：")
print(km.cluster_centers_.round(2))
counts = pd.Series(labels).value_counts().sort_index()
print(f"三个簇的样本数：{counts.to_dict()}")

# ========================================================
# 第 3 步：聚类结果可视化
# ========================================================
fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#DD8452", "#4C72B0", "#55A868"]

for c in range(3):
    sub = df[df["cluster"] == c]
    ax.scatter(
        sub["income"], sub["balance"],
        s=50, alpha=0.75,
        color=colors[c], edgecolors="white", linewidths=0.5,
        zorder=2, label=f"簇 {c}（{len(sub)} 人）",
    )

# 簇中心：大星形
ax.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    marker="*", s=800, c="red", edgecolors="black", linewidths=1.5,
    zorder=5, label="簇中心",
)

# 标注每簇人数
for c in range(3):
    cx, cy = km.cluster_centers_[c]
    n = int((df["cluster"] == c).sum())
    ax.annotate(
        f"共 {n} 人",
        xy=(cx, cy), xytext=(cx + 1.5, cy + 20000),
        fontsize=11, fontweight="bold", color=colors[c],
        arrowprops=dict(arrowstyle="->", color=colors[c], lw=1.2),
        zorder=6,
    )

ax.set_title("300 位零售银行客户 · 收入-余额聚类结果（k=3，balance 单位：元）", fontsize=14)
ax.set_xlabel("年收入 income（万元）", fontsize=12)
ax.set_ylabel("账户平均余额 balance（元）", fontsize=12)
ax.legend(fontsize=10, loc="upper left")
ax.grid(alpha=0.3, zorder=1)

plt.tight_layout()
plt.savefig("outputs/02_yuan_fail.png", dpi=150)
print("\n已保存 outputs/02_yuan_fail.png")

# ========================================================
# 第 4 步：分群报告（如实统计，余额保留"元"单位）
# ========================================================
summary = df.groupby("cluster").agg(
    人数=("customer_id", "count"),
    平均年龄=("age", "mean"),
    平均收入万元=("income", "mean"),
    平均余额元=("balance", "mean"),
    平均持有产品数=("product_count", "mean"),
).round(2)
summary["占比"] = (summary["人数"] / len(df) * 100).round(1).astype(str) + "%"

print("\n【第 4 步 · 各簇统计（balance 单位：元）】")
print(summary.to_string())

# 按平均收入排序起名（和上一轮逻辑一致）
order = summary["平均收入万元"].sort_values().index.tolist()
names = ["低收低余客群", "中收中余客群", "高收高余客群"]
biz_names = {cl: nm for cl, nm in zip(order, names)}

lines = [
    "# 客户分群分析报告（真实单位版，balance 以元计）",
    "",
    f"共分析 **{len(df)}** 位零售银行客户，特征为「年收入 income（万元）」和「账户平均余额 balance（元）」，"
    f"使用 KMeans 聚类算法（random_state=42，n_init=10），自然分为 3 个群体。",
    "",
    "## 各簇统计概览",
    "",
    summary[["人数", "占比", "平均年龄", "平均收入万元", "平均余额元", "平均持有产品数"]].to_markdown(),
    "",
    "## 逐群解读",
    "",
]
for cl in sorted(df["cluster"].unique()):
    row = summary.loc[cl]
    lines.append(
        f"- **簇 {cl} · {biz_names[cl]}**：共 **{int(row['人数'])}** 人"
        f"（占比 **{row['占比']}**），"
        f"平均年龄 **{row['平均年龄']}** 岁、平均年收入 **{row['平均收入万元']}** 万元、"
        f"平均余额 **{row['平均余额元']}** 元、平均持有 **{row['平均持有产品数']}** 个产品。"
    )

lines += [
    "",
    "## 业务命名",
    "",
]
for cl in sorted(df["cluster"].unique()):
    lines.append(f"- 簇 {cl} → **{biz_names[cl]}**")

lines += [
    "",
    "## 一句话结论",
    "",
    "300 位客户在「收入-余额」平面上自然分成三块——"
    f"**{biz_names[order[0]]}**、**{biz_names[order[1]]}**、**{biz_names[order[2]]}**。",
]

with open("generated/mall_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n已更新 generated/mall_report.md")
print("\n===== 全部完成 =====")
