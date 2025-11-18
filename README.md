# 📘 推荐系统全流程教学项目：从召回到重排  
# 📘 Full Pipeline Recommender System Tutorial: Recall → Pre-Rank → Rank → Re-Rank

---

## 🇨🇳 项目简介（中文）

本项目是一个基于 MovieLens-1M 的推荐系统完整教学示例，涵盖了工业界推荐系统的四个核心阶段：

**召回 Recall → 粗排 Pre-Rank → 精排 Rank → 重排 Re-Rank**

项目特点：

- 全流程可运行，结果可复现  
- 注释完整、结构清晰，特别适合教学与自学  
- 每个阶段均包含独立指标（Recall@K / HitRate@K / NDCG@K / AUC / ILD）  
- 模型结构贴近真实推荐系统架构  
- 提供 Google Colab Notebook（本仓库提供的 .ipynb 文件）

适用人群：

- 推荐系统初学者  
- 机器学习 / 数据科学学生  
- 算法或推荐系统岗位面试准备  
- 希望快速理解推荐架构的工程师与研究者  

---

## 🇺🇸 Project Overview (English)

This repository provides a complete, fully executable MovieLens-1M recommender system tutorial, covering all major stages used in real-world recommender systems:

**Recall → Pre-Ranking → Ranking → Re-Ranking**

Key Features:

- Fully reproducible end-to-end pipeline  
- Teaching-oriented with clear explanations and clean code  
- Metrics for each stage (Recall@K, HitRate@K, NDCG@K, AUC, ILD)  
- Mirrors real industry recommender architectures  
- Includes Google Colab notebook (.ipynb file)

Ideal For:

- Beginners learning recommender systems  
- Students in ML / Data Science  
- Interview preparation  
- Engineers & researchers needing a clean reference pipeline  

---

# 📂 仓库结构 / Repository Structure

```
rec-sys-full-pipeline/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── movielens_full_pipeline_colab.ipynb   # 主教学 Notebook
│
├── src/   # 可选：未来可将 notebook 代码模块化
│   ├── models/
│   ├── train_scripts/
│   ├── utils/
│   └── ...
│
└── data/  # 自动下载，不上传 GitHub
```

---

# 📦 安装依赖 / Installation

```
pip install -r requirements.txt
```

requirements.txt 示例：

```
torch
pandas
numpy
scikit-learn
tqdm
joblib
```

---

# 🎬 数据集 / Dataset

使用 **MovieLens-1M** 数据集。

Notebook 会自动完成：

1. 下载 `ml-1m.zip`  
2. 解压  
3. 加载 `ratings.dat`  
4. 构建训练样本 `(user_id, item_id, label)`  

评分转 label：

```
rating ≥ 4 → label = 1（正样本）
rating < 4 → label = 0（负样本）
```

---

# 🔶 阶段一：召回 Recall（DSSM 双塔模型）

召回模型结构：

- user embedding  
- item embedding  
- 点积作为匹配得分：`u · v`

训练目标：

- 正样本：点积大  
- 负样本：点积小  
- 本质是 **metric learning（度量学习）**，不是 CTR 模型

召回评估指标：

- Recall@K  
- HitRate@K  
- NDCG@K  

---

# 🔷 阶段二：粗排 Pre-Rank（小型 MLP）

输入：

- user_emb  
- item_emb  
- recall_logit  

作用：快速二次过滤召回结果

粗排指标：

- AUC  
- LogLoss  

---

# 🟦 阶段三：精排 Rank（深度 MLP）

输入：

- user_emb  
- item_emb  
- recall_logit  
- prerank_logit  

作用：拟合真实 CTR（最重要的排序模型）

精排指标：

- AUC  
- LogLoss  
- Precision@K  
- NDCG@K  

---

# 🟩 阶段四：重排 Re-Rank（MMR 多样性）

为了避免推荐结果“洗脸式相似”，使用 MMR 进行多样性重排：

```
MMR = λ * relevance  -  (1 − λ) * similarity
```

重排指标：

- ILD（Intra-list Diversity）—— 越高越多样化

---

# 🚀 端到端 Demo / End-to-End Demo

Notebook 提供完整推理：

```
recommend_for_user(
    user_id,
    topk_recall=200,
    topk_final=20,
    lambda_mmr=0.5
)
```

流程：

1. ANN 召回  
2. 粗排  
3. 精排  
4. MMR 重排  
5. 输出最终 top-N  
6. 计算多样性 ILD  

---

# 🧩 可扩展方向 / Possible Extensions

你可以继续扩展本项目，例如：

- 添加 DIN/DIEN（用户行为序列）  
- 多模态推荐（电影海报 embedding）  
- 加入 transformer / BERT4Rec  
- 图模型（GNN-based 推荐）  
- 使用 FAISS/HNSW 加速 ANN  
- 使用 BPR / InfoNCE 训练召回塔  
- 增加电影类型、tag、时间等 side features  
