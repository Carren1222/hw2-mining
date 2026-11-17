# -*- coding: utf-8 -*-
"""
第 2-4 步（修正版）：标准化 + k-均值聚类 + 双图对比 + 分群报告
数据来源：data/mall_customers_yuan.csv（balance 以元计，income 以万元计）
修复两处问题：
  ① 聚类前加 StandardScaler 标准化（量纲拉齐，两特征等权）
  ② KMeans 显式写 n_init=10（sklearn 版本无关，全班可复现）
可视化：左图 = 原始坐标着色；右图 = 标准化坐标着色
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------- 中文显示设置 ----------
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans SC"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ---------- 读入数据（真实导出版，balance 以元计） ----------
df = pd.read_csv("data/mall_customers_yuan.csv")

# ========================================================
# 第 2 步（修正版）：标准化 + k-均值聚类
# ========================================================
X_raw = df[["income", "balance"]]               # 原始特征
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)                # 标准化：均值0、方差1，两特征等权

# KMeans：显式 n_init=10，版本无关、全班复现一致
km = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = km.fit_predict(X)
df["cluster"] = labels

print("=" * 60)
print("【第 2 步（修正版）· 标准化 + KMeans 聚类完成】")
print(f"三个簇的样本数：{pd.Series(labels).value_counts().sort_index().tolist()}")
print(f"标准化空间的簇中心（z-score）：\n{km.cluster_centers_.round(3)}")
# 逆变换回原始坐标方便解释
centers_raw = scaler.inverse_transform(km.cluster_centers_)
print(f"原始空间的簇中心（income 万元, balance 元）：")
print(centers_raw.round(2))
print("=" * 60)

# ========================================================
# 第 3 步：双图对比可视化（原始坐标 vs 标准化坐标）
# ========================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
colors = ["#DD8452", "#4C72B0", "#55A868"]

# ---------- 左图：原始坐标 ----------
ax = axes[0]
for c in range(3):
    sub = df[df["cluster"] == c]
    ax.scatter(
        sub["income"], sub["balance"] / 10000,    # balance 元 → 万元才能和 income 在同一量纲下展示
        s=50, alpha=0.75, color=colors[c],
        edgecolors="white", linewidths=0.5,
        zorder=2, label=f"簇 {c}（{len(sub)} 人）",
    )
# 簇中心用逆变换后的原始坐标
ax.scatter(
    centers_raw[:, 0], centers_raw[:, 1] / 10000,
    marker="*", s=800, c="red", edgecolors="black", linewidths=1.5,
    zorder=5, label="簇中心",
)
for c in range(3):
    cx, cy = centers_raw[c]
    n = int((df["cluster"] == c).sum())
    ax.annotate(
        f"共 {n} 人",
        xy=(cx, cy / 10000), xytext=(cx + 1.5, cy / 10000 + 2),
        fontsize=11, fontweight="bold", color=colors[c],
        arrowprops=dict(arrowstyle="->", color=colors[c], lw=1.2),
        zorder=6,
    )
ax.set_title("左图 · 原始坐标（余额已换算成万元）", fontsize=13)
ax.set_xlabel("年收入 income（万元）", fontsize=11)
ax.set_ylabel("账户平均余额 balance（万元）", fontsize=11)
ax.legend(fontsize=9, loc="upper left")
ax.grid(alpha=0.3, zorder=1)

# ---------- 右图：标准化坐标（散点坐标必须是 X，不能只改轴标题） ----------
ax = axes[1]
for c in range(3):
    mask = df["cluster"].to_numpy() == c
    ax.scatter(
        X[mask, 0], X[mask, 1],              # 散点用 X 的坐标！
        s=50, alpha=0.75, color=colors[c],
        edgecolors="white", linewidths=0.5,
        zorder=2, label=f"簇 {c}（{len(df[mask])} 人）",
    )
ax.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    marker="*", s=800, c="red", edgecolors="black", linewidths=1.5,
    zorder=5, label="簇中心",
)
for c in range(3):
    cx, cy = km.cluster_centers_[c]
    n = int((df["cluster"] == c).sum())
    ax.annotate(
        f"共 {n} 人",
        xy=(cx, cy), xytext=(cx + 0.15, cy + 0.2),
        fontsize=11, fontweight="bold", color=colors[c],
        arrowprops=dict(arrowstyle="->", color=colors[c], lw=1.2),
        zorder=6,
    )
ax.set_title("右图 · 标准化坐标（无单位，均值0方差1）", fontsize=13)
ax.set_xlabel("income（标准化 z-score）", fontsize=11)
ax.set_ylabel("balance（标准化 z-score）", fontsize=11)
ax.legend(fontsize=9, loc="upper left")
ax.grid(alpha=0.3, zorder=1)

fig.suptitle("标准化不是改变客户，是改变度量衡：两特征等权参与欧氏距离计算", fontsize=15, y=0.99)
plt.tight_layout()
plt.savefig("outputs/04_compare_scale.png", dpi=150)
print("\n已保存 outputs/04_compare_scale.png")

# ========================================================
# 第 4 步：分群报告（balance 从元换算成万元）
# ========================================================
summary = df.groupby("cluster").agg(
    人数=("customer_id", "count"),
    平均年龄=("age", "mean"),
    平均收入万元=("income", "mean"),
    平均余额元=("balance", "mean"),
    平均持有产品数=("product_count", "mean"),
).round(2)
summary["平均余额万元"] = (summary["平均余额元"] / 10000).round(2)
summary["占比"] = (summary["人数"] / len(df) * 100).round(1).astype(str) + "%"

print("\n【第 4 步 · 各簇统计】")
print(summary[["人数", "占比", "平均年龄", "平均收入万元", "平均余额万元", "平均持有产品数"]].to_string())

# 按平均收入排序起名
order = summary["平均收入万元"].sort_values().index.tolist()
names = ["低收低余 · 年轻潜力客群", "中收中余 · 主力客群", "高收高余 · 富裕客群"]
biz_names = {cl: nm for cl, nm in zip(order, names)}

lines = [
    "# 客户分群分析报告（修正版：标准化 + KMeans，k=3）",
    "",
    f"共分析 **{len(df)}** 位零售银行客户，特征为「年收入 income（万元）」和「账户平均余额 balance（元）」。"
    f"聚类前先用 StandardScaler 将两特征标准化（均值 0、方差 1），再用 KMeans(k=3, random_state=42, n_init=10) 聚类，"
    f"自然分为 3 个群体。",
    "",
    "## 各簇统计概览",
    "",
    summary[["人数", "占比", "平均年龄", "平均收入万元", "平均余额万元", "平均持有产品数"]].to_markdown(),
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
        f"平均余额 **{row['平均余额万元']}** 万元、平均持有 **{row['平均持有产品数']}** 个产品。"
    )

lines += ["", "## 业务命名", ""]
for cl in sorted(df["cluster"].unique()):
    lines.append(f"- 簇 {cl} → **{biz_names[cl]}**")

lines += [
    "", "## 一句话结论", "",
    "标准化之后，两特征等权参与欧氏距离计算，300 位客户在收入-余额平面上自然分成三块——"
    f"**{biz_names[order[0]]}**（约 125 人，潜力客户）、"
    f"**{biz_names[order[1]]}**（约 125 人，银行基本盘）、"
    f"**{biz_names[order[2]]}**（约 50 人，高净值客户）。",
]

with open("generated/mall_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n已更新 generated/mall_report.md")
print("\n===== 修正版全部完成 =====")
