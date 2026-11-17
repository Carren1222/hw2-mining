# -*- coding: utf-8 -*-
"""
Step 3 · 分群画像（O9-step3，本课高潮）
故意实验 + 标准化版 并排对比，输出画像表 + 雷达图 + 分组柱状图
故意实验：不做标准化直接聚类——balance(0~72万) 碾压一切
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------- 中文显示设置 ----------
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans SC"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ---------- 读入数据 ----------
df = pd.read_csv("data/mall_customers_yuan.csv")

# 4 个特征（故意实验：用原始 balance 元，不做任何单位换算）
FEATURES_RAW = ["income", "balance", "age", "product_count"]
# 画像展示时 balance 换成万元方便读
df["balance_wan"] = df["balance"] / 10000
FEATURES_PROFILE = ["income", "balance_wan", "age", "product_count"]

# 故意实验里各特征的量级
print("=" * 70)
print("【Step 3 · 分群画像】")
print(f"4 个特征的量级（故意实验会直接喂给 KMeans，不做标准化）：")
for col in FEATURES_RAW:
    s = df[col]
    print(f"  {col:15s}: min={s.min():>10.2f}, max={s.max():>10.2f}, "
          f"量级差≈ {s.max()/s.min() if s.min()!=0 else 'inf':>10}")
print("=" * 70)

# ========================================================
# 故意实验 · 版本 A：不做标准化，原始 balance 直接上
# ========================================================
print("\n【故意实验 · 版本 A】不做标准化，balance(元) 直接喂 KMeans")
print(f"  → balance(0~720000) 碾压 income(2~53)、age(18~63)、product_count(0~8)")

X_A = df[FEATURES_RAW].values
km_A = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster_A"] = km_A.fit_predict(X_A)
counts_A = df["cluster_A"].value_counts().sort_index()
print(f"  各簇人数：{dict(counts_A)}")
print(f"  比例分布：{dict((c, f'{n/len(df)*100:.1f}%') for c, n in counts_A.items())}")

# 画像用原始单位（balance 换成万元好读）
画像_A = df.groupby("cluster_A")[FEATURES_PROFILE].mean().round(2)
画像_A.index.name = "簇"
print("\n  画像（各簇特征均值）：")
print(画像_A.to_string())

# ========================================================
# 正确做法 · 版本 B：先标准化再聚类
# ========================================================
print("\n【正确做法 · 版本 B】先 StandardScaler，再 KMeans，量纲拉齐")

scaler = StandardScaler()
X_B = scaler.fit_transform(X_A)
km_B = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster_B"] = km_B.fit_predict(X_B)
counts_B = df["cluster_B"].value_counts().sort_index()
print(f"  各簇人数：{dict(counts_B)}")
print(f"  比例分布：{dict((c, f'{n/len(df)*100:.1f}%') for c, n in counts_B.items())}")

画像_B = df.groupby("cluster_B")[FEATURES_PROFILE].mean().round(2)
画像_B.index.name = "簇"
print("\n  画像（各簇特征均值）：")
print(画像_B.to_string())

# ========================================================
# 给版本 B 的 3 个簇起业务名字（按画像解读）
# ========================================================
print("\n" + "=" * 70)
print("【业务命名】（基于版本 B 的完整画像）")
biz_names_B = {}
# 先打印每个簇的完整画像，再起名
print("\n  各簇画像详情：")
for cl in sorted(df["cluster_B"].unique()):
    r = 画像_B.loc[cl]
    print(f"    簇 {cl}: 收入={r['income']:.1f}万, 余额={r['balance_wan']:.1f}万, "
          f"年龄={r['age']:.0f}岁, 产品数={r['product_count']:.1f}")

# 排序起名：按收入从低到高
order = 画像_B["income"].sort_values().index.tolist()
biz_names_B[order[0]] = "低收低余 · 年轻潜力客群"    # 收入最低、余额最低、年龄最轻
biz_names_B[order[1]] = "中收中余 · 主力中产客群"    # 中间档、产品数最多（最活跃）
biz_names_B[order[2]] = "高收高余 · 富裕稳健客群"    # 收入最高、余额最高、年龄最大

print(f"\n  命名方案：")
for cl in sorted(df["cluster_B"].unique()):
    print(f"    簇 {cl} → {biz_names_B[cl]}")

# ========================================================
# 对比结论
# ========================================================
print("\n" + "=" * 70)
print("【对比结论】")
print(f"  故意实验（不标准化）：簇人数 = {counts_A.tolist()}  → 余额单一维度主导分群")
print(f"  正确做法（标准化）：  簇人数 = {counts_B.tolist()}  → 4 特征等权，均衡分群")
print("=" * 70)

# ========================================================
# 可视化 1：版本 B 的画像 · 分组柱状图 + 雷达图
# ========================================================
def minmax_norm(series):
    return (series - series.min()) / (series.max() - series.min())

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 归一化画像（跨簇做 minmax，方便同图对比）
norm_B = 画像_B.copy()
for col in norm_B.columns:
    norm_B[col] = minmax_norm(norm_B[col])

colors = ["#DD8452", "#4C72B0", "#55A868"]

# 左图：分组柱状图
ax = axes[0]
x = np.arange(len(画像_B.columns))
width = 0.22
for i, cl in enumerate(sorted(df["cluster_B"].unique())):
    vals = norm_B.loc[cl].values
    short_name = biz_names_B[cl].split(" · ")[0]
    bars = ax.bar(x + i * width, vals, width,
                  label=f"簇 {cl} · {short_name}",
                  color=colors[i])
    for bar, v in zip(bars, vals):
        ax.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7, color="#333")

ax.set_xticks(x + width)
ax.set_xticklabels(["年收入(万)", "账户余额(万)", "年龄(岁)", "持有产品数"], fontsize=10)
ax.set_ylabel("归一化（0=簇内最低, 1=簇内最高）", fontsize=11)
ax.set_title("左图 · 分组柱状图（4 特征等权对比）", fontsize=13)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, linestyle="--", axis="y")

# 右图：雷达图
ax = axes[1]
labels = ["年收入", "账户余额", "年龄", "持有产品数"]
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

for i, cl in enumerate(sorted(df["cluster_B"].unique())):
    vals = norm_B.loc[cl].values.tolist()
    vals += vals[:1]
    short_name = biz_names_B[cl].split(" · ")[0]
    ax.plot(angles, vals, "o-", linewidth=2,
            label=f"簇 {cl} · {short_name}",
            color=colors[i])
    ax.fill(angles, vals, alpha=0.15, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=11)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, color="#666")
ax.set_title("右图 · 雷达图（越外圈该特征越突出）", fontsize=13)
ax.legend(fontsize=9, loc="upper right", bbox_to_anchor=(1.35, 1.1))

fig.suptitle("Step 3 · 分群画像（k=3，标准化后，4 特征等权）", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("outputs/03b_cluster_profile.png", dpi=150, bbox_inches="tight")
print("\n✅ 已保存 outputs/03b_cluster_profile.png")

# ========================================================
# 可视化 2：故意实验 vs 正确做法 · 双雷达并排
# ========================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for side, (ax, 画像, title) in enumerate([
    (axes[0], 画像_A, "故意实验 · 不标准化"),
    (axes[1], 画像_B, "正确做法 · 标准化后"),
]):
    normed = 画像.copy()
    for col in normed.columns:
        normed[col] = minmax_norm(normed[col])

    labels = ["年收入(万)", "账户余额(万)", "年龄(岁)", "持有产品数"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    counts = counts_A if side == 0 else counts_B

    for i, cl in enumerate(sorted(画像.index)):
        vals = normed.loc[cl].values.tolist()
        vals += vals[:1]
        n = counts.loc[cl]
        ax.plot(angles, vals, "o-", linewidth=2,
                label=f"簇 {cl}（{n}人）",
                color=colors[i])
        ax.fill(angles, vals, alpha=0.15, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, color="#666")
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, loc="upper right", bbox_to_anchor=(1.35, 1.1))

fig.suptitle("故意实验：不标准化时余额(0~72万)碾压其他三特征 → 分群退化为\"按余额切三段\"",
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("outputs/03c_compare_standardization.png", dpi=150, bbox_inches="tight")
print("✅ 已保存 outputs/03c_compare_standardization.png")

print("\n===== Step 3 · 分群画像 全部完成 =====")
