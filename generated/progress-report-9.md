# 进度报告·第 9 次课：聚类客户分群

> 学号：20251030106  姓名：赵章君  日期：11/17  ｜  lesson09-课堂演示包

---

## 1. 检查点完成情况

| 检查点 | 状态 | 结果 |
|---|---|---|
| Step 1 · 聚类跑通 | ✅ | 标准化 + KMeans k=3，簇人数 126 / 124 / 50 |
| Step 2 · 肘部法则选 k | ✅ | k=3 处下降率腰斩（-57.5% → -28.1%），边际收益骤降 |
| Step 3 · 分群画像 | ✅ | 画像表 + 雷达图 + 分组柱状图 + 三个中文业务命名 |

## 2. 遇到的问题与解决方式

| 问题 | 怎么解决的 |
|---|---|
| **量纲陷阱**：balance 以元计(0~72万)，income 以万元计(2~53)，直接跑 KMeans 得到 141/48/111 | 用 StandardScaler 拉成 z-score 后重跑 → 126/124/50；额外做双图对比 |
| **sklearn 版本差异**：1.4 以后 n_init 默认变 auto，全班复现不一致 | 显式写 `n_init=10` + `random_state=42` |
| **matplotlib API 变更**：`plt.cm.get_cmap` 被移除 | 改成 `plt.get_cmap()`，一行修复 |

## 3. 自主实践完成度自查

| 任务 | 完成 | 备注 |
|---|---|---|
| 数据初检 + 初始散点图 | ☑ | outputs/01_raw_scatter.png |
| KMeans 聚类 + 簇编号写回 | ☑ | k=3, random_state=42, n_init=10 |
| 聚类着色 + 星形质心 + 人数标注 | ☑ | outputs/03_clustered_scatter.png |
| 肘部法 + 轮廓系数选 k | ☑ | outputs/05_choose_k.png（k=3 vs k=2 的权衡） |
| 手写 KMeans++ 迭代过程 | ☑ | outputs/06_iteration.png，4 轮收敛 |
| 分群画像 + 雷达图 + 柱状图 | ☑ | outputs/03b_cluster_profile.png |
| 故意实验：不标准化 vs 标准化 | ☑ | outputs/03c_compare_standardization.png |
| 加分综合任务（4 项） | ☑ | outputs/03d_multi_projection.png + 07_kmeans_fail_vs_dbscan.png |

## 4. 独立研究任务

**（1）环形分布玩具数据 → KMeans 失败，DBSCAN 成功**
- 用 `make_circles` 生成内外两圈，KMeans(k=2) 强行切两半，DBSCAN(eps=0.2) 完美识别两圈 + 标记噪声点
- 核心差异：KMeans 假设球状簇 + 欧氏距离；DBSCAN 假设密度可达 + 不限形状

**（2）特征冗余：加入 account_age_months**
- 构造 account_age_months = (age-18)×12 + 扰动，Pearson r = **0.9991**（≈ age 复制）
- 加入后簇边界轻微漂移，但三群本质不变 → 相关性 > 0.9 的特征不要同时进聚类

**（3）簇稳定性：random_state 换 4 个值**
- rs=[0,42,123,2024]，排序后簇人数和 income 均值**完全一致**
- 数据本身有明确 3 簇结构 + k-means++ 初始化质量好 → 结果稳定

## 5. AI 使用记录摘要

| 指令原文 | 我怎么验证的 | 结论 |
|---|---|---|
| 换特征集重新画像 | 特征 4→5，对比两版簇均值 | r=0.9991 冗余，解读不变 |
| random_state 换值测试 | 4 个值逐个跑 | 完全稳定，数据有自然结构 |
| 环形数据 + DBSCAN | make_circles + 三图并排 | KMeans 切两半失败，DBSCAN 成功 |

## 6. 本课一句话收获

> **KMeans 不是万能的**——它假设球状簇 + 欧氏距离，量纲差太大、数据有环形、特征高度冗余都会跑偏。**标准化 + 特征筛选 + DBSCAN 应对任意形状**，这三件事决定了聚类能不能落地。

## 7. 给教师的留言

无
