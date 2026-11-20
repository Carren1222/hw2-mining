# -*- coding: utf-8 -*-
"""
大作业补充建模：在"聚类分群"基础上再加两种模型，凑齐 分类/聚类/异常 三类
  建模二 · 分类：用聚类标签当 y，训练决策树 → 新客户自动归群
  建模三 · 异常：IsolationForest → 标记余额/收入明显背离的可疑客户
聚类沿用主线结果：标准化(income, balance) + KMeans k=3 → 126/124/50
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report, silhouette_score)
from sklearn.ensemble import IsolationForest

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans SC"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ========================================================
# 数据 + 预处理（与大作业主线完全一致）
# ========================================================
df = pd.read_csv("data/mall_customers_yuan.csv")
df["balance_wan"] = df["balance"] / 10000          # 元 → 万元
FEAT4 = ["income", "balance_wan", "age", "product_count"]

print(f"数据规模：{df.shape[0]} 行 × {df.shape[1]} 列，缺失值总数 = {df.isna().sum().sum()}")

scaler2 = StandardScaler()
X2 = scaler2.fit_transform(df[["income", "balance"]].values)      # 聚类用 2 特征
scaler4 = StandardScaler()
X4 = scaler4.fit_transform(df[FEAT4].values)                      # 分类/异常用 4 特征

# ========================================================
# 建模一 · 聚类（复现主线：KMeans k=3）
# ========================================================
km = KMeans(n_clusters=3, random_state=42, n_init=10)
y_seg = km.fit_predict(X2)                    # 聚类标签，兼作分类任务的 y
df["cluster"] = y_seg
counts = pd.Series(y_seg).value_counts().sort_index()
sil = silhouette_score(X2, y_seg)
print("\n【建模一 · 聚类】簇人数 =", counts.tolist(),
      f" 轮廓系数(k=3) = {sil:.3f}")

# 按收入排序固定业务命名：0=潜力 1=富裕 2=主力
order = df.groupby("cluster")["income"].mean().sort_values().index.tolist()
seg_name = {order[0]: "潜力客群", order[1]: "主力客群", order[2]: "富裕客群"}
print("业务命名：", {k: v for k, v in seg_name.items()})

# ========================================================
# 建模二 · 分类：决策树学习"自动归群"
# ========================================================
X_tr, X_te, y_tr, y_te = train_test_split(
    X4, y_seg, test_size=0.3, stratify=y_seg, random_state=42)

tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_tr, y_tr)
y_pred = tree.predict(X_te)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_tr, y_tr)
y_pred_lr = lr.predict(X_te)

cv_tree = cross_val_score(DecisionTreeClassifier(max_depth=4, random_state=42),
                          X4, y_seg, cv=5)
cv_lr = cross_val_score(LogisticRegression(max_iter=1000, random_state=42),
                        X4, y_seg, cv=5)

print("\n【建模二 · 分类】测试集(90人) = 30% 分层抽样, random_state=42")
print(f"  决策树  : acc={accuracy_score(y_te, y_pred):.3f}  macro-F1={f1_score(y_te, y_pred, average='macro'):.3f}"
      f"  5折CV acc={cv_tree.mean():.3f} ± {cv_tree.std():.3f}")
print(f"  逻辑回归: acc={accuracy_score(y_te, y_pred_lr):.3f}  macro-F1={f1_score(y_te, y_pred_lr, average='macro'):.3f}"
      f"  5折CV acc={cv_lr.mean():.3f} ± {cv_lr.std():.3f}")
print("\n  决策树分类报告（按簇编号）：")
print(classification_report(y_te, y_pred, digits=3))
cm = confusion_matrix(y_te, y_pred)
print("  混淆矩阵（行=真实簇 0/1/2，列=预测簇）：")
print(cm)
print("  决策树特征重要性：")
for f, w in sorted(zip(FEAT4, tree.feature_importances_), key=lambda t: -t[1]):
    print(f"    {f:15s} {w:.3f}")

# ========================================================
# 建模三 · 异常检测：IsolationForest
# ========================================================
iso = IsolationForest(n_estimators=100, contamination=0.03, random_state=42)
flag = iso.fit_predict(X4)                    # -1 = 异常, 1 = 正常
df["anomaly"] = flag
anom = df[flag == -1].copy()
anom["score"] = iso.decision_function(X4)[flag == -1]
anom["b_over_i"] = anom["balance_wan"] / anom["income"].clip(lower=0.1)

print("\n【建模三 · 异常检测】IsolationForest(contamination=0.03) → 标记",
      len(anom), "位异常客户")
print(anom[["customer_id", "age", "income", "balance_wan", "product_count",
            "score", "b_over_i"]].round(2).to_string(index=False))
normal_ratio = (df[flag == 1]["balance_wan"] / df[flag == 1]["income"].clip(lower=0.1))
print(f"  正常客户 余额/收入 中位数 = {normal_ratio.median():.2f}，"
      f"异常客户中位数 = {anom['b_over_i'].median():.2f}")

# ========================================================
# 图 1：分类评估（混淆矩阵 + 特征重要性 + 决策区域）
# ========================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
colors = ["#DD8452", "#4C72B0", "#55A868"]

ax = axes[0]
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels([f"预测{c}\n{seg_name[c]}" for c in range(3)], fontsize=9)
ax.set_yticklabels([f"真实{c}\n{seg_name[c]}" for c in range(3)], fontsize=9)
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                color="white" if cm[i, j] > cm.max() / 2 else "#333")
ax.set_title(f"(a) 混淆矩阵（测试集 acc={accuracy_score(y_te, y_pred):.2f}）", fontsize=12)

ax = axes[1]
imp = tree.feature_importances_
bars = ax.barh([f for f in FEAT4], imp, color="#4C72B0", alpha=0.85)
for b, v in zip(bars, imp):
    ax.annotate(f"{v:.2f}", (b.get_width(), b.get_y() + b.get_height() / 2),
                xytext=(4, 0), textcoords="offset points", va="center", fontsize=9)
ax.set_title("(b) 决策树特征重要性（收入+余额主导，符合业务直觉）", fontsize=11)

ax = axes[2]
xx, yy = np.meshgrid(np.linspace(X4[:, 0].min() - 0.5, X4[:, 0].max() + 0.5, 300),
                     np.linspace(X4[:, 1].min() - 0.5, X4[:, 1].max() + 0.5, 300))
age_med, pc_med = np.median(X4[:, 2]), np.median(X4[:, 3])
grid = np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, age_med), np.full(xx.size, pc_med)]
zz = tree.predict(grid).reshape(xx.shape)
ax.contourf(xx, yy, zz, alpha=0.18, levels=[-0.5, 0.5, 1.5, 2.5],
            colors=colors)
for c in range(3):
    m = y_te == c
    ax.scatter(X_te[m, 0], X_te[m, 1], s=35, alpha=0.8, color=colors[c],
               edgecolors="white", linewidths=0.5, label=f"测试集·簇{c}")
ax.set_xlabel("income（标准化）"); ax.set_ylabel("balance（标准化）")
ax.set_title("(c) 决策区域（年龄/产品数固定为中位数）", fontsize=11)
ax.legend(fontsize=8)

fig.suptitle("建模二 · 分类：决策树把聚类规则学成可复用的自动归群器（教学演示虚构数据）",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("outputs/08_classification_eval.png", dpi=150, bbox_inches="tight")
print("\n✅ 已保存 outputs/08_classification_eval.png")

# ========================================================
# 图 2：异常检测
# ========================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))

ax = axes[0]
normal = df[flag == 1]
ax.scatter(normal["income"], normal["balance_wan"], s=40, alpha=0.6,
           color="#4C72B0", edgecolors="white", linewidths=0.4, label=f"正常客户 {len(normal)} 人")
ax.scatter(anom["income"], anom["balance_wan"], s=130, marker="X",
           color="#C44E52", edgecolors="black", linewidths=0.8, zorder=5,
           label=f"异常客户 {len(anom)} 人")
for _, r in anom.iterrows():
    ax.annotate(f"#{int(r['customer_id'])}", (r["income"], r["balance_wan"]),
                xytext=(6, 4), textcoords="offset points", fontsize=8, color="#C44E52")
ax.set_xlabel("年收入 income（万元）"); ax.set_ylabel("账户余额 balance（万元）")
ax.set_title("(a) 收入-余额平面：红叉 = IsolationForest 标记的异常客户", fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=0.3, linestyle="--")

ax = axes[1]
scores = iso.decision_function(X4)
ax.hist(scores, bins=30, color="#4C72B0", alpha=0.8, edgecolor="white")
thr = np.sort(scores)[len(anom) - 1]
ax.axvline(thr, color="#C44E52", linestyle="--", linewidth=2,
           label=f"前 3% 分界（score={thr:.3f}）")
ax.hist(scores[flag == -1], bins=12, color="#C44E52", alpha=0.9,
        edgecolor="white", label="被判为异常")
ax.set_xlabel("异常评分 decision_function（越小越异常）")
ax.set_ylabel("客户数")
ax.set_title("(b) 异常评分分布：左尾 3% 被标记", fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=0.3, linestyle="--")

fig.suptitle("建模三 · 异常检测：IsolationForest 自动找出余额与收入明显背离的客户（教学演示虚构数据）",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("outputs/09_anomaly_detection.png", dpi=150, bbox_inches="tight")
print("✅ 已保存 outputs/09_anomaly_detection.png")
print("\n===== 大作业补充建模全部完成 =====")
