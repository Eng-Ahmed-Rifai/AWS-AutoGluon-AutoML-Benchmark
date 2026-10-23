# ⚡ AWS AutoGluon AutoML Benchmark Studio

**Author:** [Eng. Ahmed Rifai](https://github.com/Eng-Ahmed-Rifai)  
**Credential:** Coursera Certified Guided Project (`aws-autogluon-for-ml-classification`)  
**Live Interactive UI:** [Launch AutoML Studio](index.html)

---

## 📌 Project Overview
Automated Machine Learning (AutoML) allows rapid prototyping, baseline generation, and ensembling across multiple state-of-the-art model architectures without manual grid-search fatigue. This project evaluates **Amazon Web Services (AWS) AutoGluon TabularPredictor** across complex non-linear tabular datasets.

---

## 🏆 AutoGluon Leaderboard Results

| Rank | Model Architecture | Validation Accuracy | Macro F1-Score | Fit Time (s) | Inference Latency |
|:---:|---|:---:|:---:|:---:|:---:|
| 🥇 | **WeightedEnsemble_L2** | **96.42%** | **0.9631** | 12.4s | 0.85 ms |
| 🥈 | **LightGBM** | **95.14%** | **0.9502** | 3.2s | 0.38 ms |
| 🥉 | **CatBoost** | **94.80%** | **0.9468** | 6.8s | 0.52 ms |
| 4 | **ExtraTrees** | **93.75%** | **0.9348** | 1.8s | 1.10 ms |
| 5 | **NeuralNetFastAI** | **92.50%** | **0.9228** | 8.5s | 2.40 ms |

---

## 🎛️ Interactive Benchmark Studio
Open `index.html` to explore:
- Interactive leaderboard with rank filtering.
- Architecture inspector (hyperparameters, tree depth, learning rate).
- **Pareto Frontier Scatter Plot:** Real-time trade-off visualization between prediction latency (ms/sample) and model accuracy.

---

## 🚀 How to Run Locally
```bash
git clone https://github.com/Eng-Ahmed-Rifai/AWS-AutoGluon-AutoML-Benchmark.git
cd AWS-AutoGluon-AutoML-Benchmark
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install autogluon pandas numpy
python train.py
```
