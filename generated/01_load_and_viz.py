# -*- coding: utf-8 -*-
"""
第 1 步：数据初检与初始可视化
读入 mall_customers_sim.csv，打印基本信息，
绘制「收入-余额」散点图（点大小=年龄），保存到 outputs/01_raw_scatter.png
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# ---------- 中文显示设置 ----------
# Windows 上可用的中文字体优先级：SimHei > Microsoft YaHei > Noto Sans SC
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans SC", "KaiTi"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ---------- 读入数据 ----------
df = pd.read_csv("data/mall_customers_sim.csv")

# ---------- 数据初检 ----------
print("=" * 55)
print("【数据初检】")
print(f"行列数：{df.shape[0]} 行 × {df.shape[1]} 列")
print("\n各列缺失值：")
print(df.isna().sum())
print("\n数值列基本统计（保留 2 位小数）：")
print(df.describe().round(2))
print("=" * 55)

# ---------- 初始散点图：收入-余额 ----------
fig, ax = plt.subplots(figsize=(10, 7))

# 点的大小用 age 映射：最年轻 18 岁 → 20px，最年长 63 岁 → 360px
point_size = (df["age"] - df["age"].min()) * 8 + 20

ax.scatter(
    df["income"], df["balance"],
    s=point_size,
    alpha=0.6,
    c="#4C72B0",
    edgecolors="white",
    linewidths=0.5,
    zorder=2,
)

ax.set_title("300 位零售银行客户：收入-余额分布（点越大 = 年龄越高）", fontsize=15)
ax.set_xlabel("年收入 income（万元）", fontsize=12)
ax.set_ylabel("账户平均余额 balance（万元）", fontsize=12)
ax.grid(alpha=0.3, zorder=1)

plt.tight_layout()
plt.savefig("outputs/01_raw_scatter.png", dpi=150)
print("\n已保存 outputs/01_raw_scatter.png")
