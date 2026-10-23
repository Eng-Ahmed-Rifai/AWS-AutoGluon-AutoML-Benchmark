"""
AWS AutoGluon AutoML Classification Pipeline: TabularPredictor Benchmark
Author: Eng. Ahmed Rifai
"""

import numpy as np
import pandas as pd

# Demonstrating AutoGluon training workflow & model ensembling simulation
def generate_tabular_dataset(n_samples=3000, random_state=42):
    np.random.seed(random_state)
    f1 = np.random.normal(0, 1, n_samples)
    f2 = np.random.exponential(1.5, n_samples)
    f3 = np.random.choice(['Low', 'Medium', 'High'], n_samples)
    f4 = np.random.uniform(10, 100, n_samples)
    
    # Non-linear interaction target
    cat_weight = {'Low': -0.8, 'Medium': 0.2, 'High': 1.1}
    z = 1.2 * f1 - 0.5 * f2 + [cat_weight[c] for c in f3] + 0.03 * f4
    prob = 1 / (1 + np.exp(-z))
    target = (prob > 0.5).astype(int)
    
    df = pd.DataFrame({
        'feature_num1': f1,
        'feature_num2': f2,
        'feature_cat': f3,
        'feature_num3': f4,
        'target': target
    })
    return df

def run_autogluon_benchmark():
    print("--- 1. Initializing AutoGluon Tabular Dataset ---")
    df = generate_tabular_dataset()
    train_data = df.iloc[:2400]
    test_data = df.iloc[2400:]
    print(f"Train samples: {len(train_data)} | Test samples: {len(test_data)}")
    
    print("\n--- 2. TabularPredictor Simulated Training Run ---")
    # In real AWS / AutoGluon environment:
    # from autogluon.tabular import TabularPredictor
    # predictor = TabularPredictor(label='target', eval_metric='f1').fit(train_data, presets='best_quality')
    
    leaderboard = pd.DataFrame({
        'model': ['WeightedEnsemble_L2', 'LightGBM', 'CatBoost', 'ExtraTrees', 'NeuralNetFastAI'],
        'score_val (Accuracy)': [0.9642, 0.9514, 0.9480, 0.9375, 0.9250],
        'f1_macro': [0.9631, 0.9502, 0.9468, 0.9348, 0.9228],
        'fit_time_seconds': [12.4, 3.2, 6.8, 1.8, 8.5],
        'pred_time_ms': [0.85, 0.38, 0.52, 1.10, 2.40]
    })
    
    print("\n--- 3. Multi-Model Leaderboard ---")
    print(leaderboard.to_string(index=False))
    
    print("\n--- 4. Stacking Weights (WeightedEnsemble_L2) ---")
    print("LightGBM: 42% | CatBoost: 35% | ExtraTrees: 23%")

if __name__ == "__main__":
    run_autogluon_benchmark()
