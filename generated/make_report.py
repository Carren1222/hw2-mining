# -*- coding: utf-8 -*-
"""按照 aip-progress-report-9.docx 模板填写实验报告"""
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()

# 设置中文字体
normal = doc.styles['Normal']
normal.font.name = 'Microsoft YaHei'
normal.font.size = Pt(11)

# ========== 标题 ==========
title = doc.add_heading("进度报告·第 9 次课：聚类客户分群", level=1)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

# ========== 基本信息 ==========
doc.add_paragraph("学号：＿＿＿＿  姓名：＿＿＿＿  日期：11/17  ｜  lesson09-课堂演示包")
doc.add_paragraph()

# ========== 1. 检查点完成情况 ==========
doc.add_heading("1. 检查点完成情况", level=2)
doc.add_paragraph("Step 1 · 聚类跑通（tag O9-step1）：✅ 完成，簇人数 126 / 124 / 50（标准化 + KMeans k=3）")
doc.add_paragraph("Step 2 · 肘部法则选 k（tag O9-step2）：✅ 完成，k=3 处下降率从 -57.5% 骤降到 -28.1%，边际收益腰斩")
doc.add_paragraph("Step 3 · 分群画像（tag O9-step3，本课高潮）：✅ 完成，画像表 + 雷达图 + 分组柱状图 + 三中文业务命名")
doc.add_paragraph()

# ========== 2. 遇到的问题与解决方式 ==========
doc.add_heading("2. 遇到的问题与解决方式（至少 1 条）", level=2)

table = doc.add_table(rows=4, cols=2)
table.style = 'Table Grid'
headers = ["问题", "怎么解决的（AI 怎么帮的 / 查了什么 / 自己试了什么）"]
for i, h in enumerate(headers):
    table.rows[0].cells[i].text = h
    for p in table.rows[0].cells[i].paragraphs:
        for run in p.runs:
            run.font.bold = True

problems = [
    ("量纲陷阱：balance 以元计(0~72万)，income 以万元计(2~53)，直接跑 KMeans 得到 141/48/111 而非预期的 125/125/50",
     "① 查 KMeans 文档，确认欧氏距离里数值大的列会主导结果；② 用 StandardScaler 把两特征都拉成 z-score（均值0、方差1），再跑一遍 → 126/124/50，量纲拉齐了；③ 额外做了双图对比（原始坐标 vs 标准化坐标并排），一眼看懂标准化的作用。"),
    ("sklearn 1.4 以后 KMeans 的 n_init 默认值变成 auto，不同机器跑出来簇编号可能不一样",
     "显式写 n_init=10 + random_state=42，全班复现一致。稳定性测试换了 0/42/123/2024 四个 random_state，排序后的簇人数和各簇均值完全一致，说明 k-means++ 初始化质量好。"),
    ("plt.cm.get_cmap 在新版 matplotlib 里被移除，加分④跑 DBSCAN 对比时报 AttributeError",
     "改成 plt.get_cmap()（3.7+ 的新 API），一行修复。"),
]
for i, (q, s) in enumerate(problems, 1):
    table.rows[i].cells[0].text = q
    table.rows[i].cells[1].text = s

doc.add_paragraph()

# ========== 3. 自主实践各任务完成度自查 ==========
doc.add_heading("3. 自主实践各任务完成度自查", level=2)

table2 = doc.add_table(rows=9, cols=3)
table2.style = 'Table Grid'

rows_data = [
    ("任务", "完成", "备注"),
    ("数据初检与初始散点图", "☑", "outputs/01_raw_scatter.png，300 行 7 列无缺失"),
    ("KMeans 聚类（income + balance）", "☑", "簇编号写回 df['cluster']，k=3，random_state=42"),
    ("聚类结果着色 + 簇中心星形 + 人数标注", "☑", "outputs/03_clustered_scatter.png"),
    ("肘部法 + 轮廓系数选 k", "☑", "outputs/05_choose_k.png，两指标分别指向 k=3 和 k=2，权衡后选 k=3"),
    ("手写 KMeans++ 迭代过程可视化", "☑", "outputs/06_iteration.png，2×2 快照，4 轮收敛"),
    ("分群画像：均值表 + 雷达图 + 分组柱状图", "☑", "outputs/03b_cluster_profile.png，三个中文业务命名"),
    ("故意实验：不标准化 vs 标准化 画像对比", "☑", "outputs/03c_compare_standardization.png，量纲陷阱真实可证"),
    ("加分综合任务（4 项）", "☑", "outputs/03d_multi_projection.png + 07_kmeans_fail_vs_dbscan.png"),
]

for i, row_data in enumerate(rows_data):
    for j, val in enumerate(row_data):
        table2.rows[i].cells[j].text = val
        for p in table2.rows[i].cells[j].paragraphs:
            for run in p.runs:
                run.font.size = Pt(10)
                if i == 0:
                    run.font.bold = True

doc.add_paragraph()

# ========== 4. 独立研究任务 ==========
doc.add_heading("4. 独立研究任务（做了的写）", level=2)

doc.add_paragraph("（1）环形分布玩具数据验证 KMeans 失败 + DBSCAN 成功", style='List Bullet')
doc.add_paragraph("→ 我的做法：用 sklearn.datasets.make_circles 生成内外两圈玩具数据，同时跑 KMeans(k=2) 和 DBSCAN(eps=0.2)。")
doc.add_paragraph("→ 结论：KMeans 强行切两半（把内外各劈 50%），DBSCAN 完美识别两圈 + 自动标记噪声点。核心差异：KMeans 假设球状簇 + 欧氏距离，DBSCAN 假设密度可达 + 不设形状。图见 outputs/07_kmeans_fail_vs_dbscan.png。")

doc.add_paragraph("（2）特征冗余实验：加入 account_age_months", style='List Bullet')
doc.add_paragraph("→ 我的做法：构造 account_age_months = (age - 18) × 12 + 小扰动，Pearson r = 0.9991，等于 age 的冗余特征。加入后 KMeans 分群人数从 [111,138,51] 变成 [145,104,51]，但按 income 排序后的均值几乎一模一样。")
doc.add_paragraph("→ 结论：相关性 > 0.9 的特征不要同时进聚类——会造成某维度权重翻倍、簇边界轻微漂移，但三群的本质结构不变（年轻 / 中产 / 富裕）。选特征的原则：业务可解释性 > 统计独立性 > 维度数少。")

doc.add_paragraph("（3）簇稳定性测试：random_state 换 4 个值", style='List Bullet')
doc.add_paragraph("→ 我的做法：random_state 取 [0, 42, 123, 2024]，排序后簇人数完全一致 [138, 111, 51]，income 均值完全相同 [7.16, 18.80, 41.61]。")
doc.add_paragraph("→ 结论：数据本身有明确 3 簇自然结构（不是 KMeans 硬切），加上 k-means++ 初始化质量好，分群非常稳定。但如果改用 init='random'（老版本默认），稳定性会差得多。")

doc.add_paragraph()

# ========== 5. AI 使用记录摘要 ==========
doc.add_heading("5. AI 使用记录摘要", level=2)

ai_table = doc.add_table(rows=4, cols=3)
ai_table.style = 'Table Grid'

ai_headers = ["指令原文（关键 1-3 条）", "我怎么验证的", "结论"]
for i, h in enumerate(ai_headers):
    ai_table.rows[0].cells[i].text = h
    for p in ai_table.rows[0].cells[i].paragraphs:
        for run in p.runs:
            run.font.bold = True
            run.font.size = Pt(11)

ai_rows = [
    ("换特征集（加入 account_age_months）重新画像",
     "脚本 06_bonus_challenges.py 里特征 4→5，对比两版簇人数和画像均值",
     "r=0.9991 高度冗余，加入后簇边界轻微漂移但业务解读不变"),
    ("random_state 换 3 个值，画像是否大变",
     "rs=[0,42,123,2024] 逐个跑，排序后簇人数和 income 均值全一样",
     "分群非常稳定，数据本身有明确 3 簇结构"),
    ("生成环形玩具数据验证 KMeans 失败 + DBSCAN 对比",
     "make_circles + KMeans(k=2) + DBSCAN(eps=0.2)，画并排三图",
     "KMeans 切两半失败，DBSCAN 完美识别两圈。形状假设 vs 密度假设的本质差异"),
]

for i, row_data in enumerate(ai_rows, 1):
    for j, val in enumerate(row_data):
        ai_table.rows[i].cells[j].text = val
        for p in ai_table.rows[i].cells[j].paragraphs:
            for run in p.runs:
                run.font.size = Pt(10)

doc.add_paragraph()

# ========== 6. 本课一句话收获 ==========
doc.add_heading("6. 本课一句话收获", level=2)

quote = doc.add_paragraph()
quote.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = quote.add_run(
    "KMeans 不是'万能的'——它假设球状簇 + 欧氏距离，这两个前提一旦不成立"
    "（量纲差太大、数据有环形/弯带、特征高度冗余），结果就会跑偏。"
    "标准化选 k、特征筛选、用 DBSCAN 应对任意形状——这三件事决定了聚类能不能落地。"
)
run.font.italic = True
run.font.size = Pt(12)

doc.add_paragraph()

# ========== 7. 给教师的留言 ==========
doc.add_heading("7. 给教师的留言（可选）", level=2)

doc.add_paragraph("① 加分①里 account_age_months 和 age 高度冗余（r≈1.0），是不是数据生成时故意做的？如果是，那非常好——等于用一道题同时考了'特征筛选'这个考点。")
doc.add_paragraph("② 故意实验里 4 特征版和 2 特征版的簇编号不一样（111/138/50 vs 126/124/50），但按画像解读后三个客群的本质结构是一样的（年轻/中产/富裕）。要不要在讲解里提一句——特征越多，簇边界可能微调，但大画像不变？")

# ========== 保存 ==========
output_path = r"d:\九酒yang\Desktop\TRAE_Code\lesson09-课堂演示包\lesson09-课堂演示包\generated\progress-report-9.docx"
doc.save(output_path)
print(f"✅ 已保存 {output_path}")
