# Homework Assignment: Criteo Click-Through Rate Prediction Competition

## Overview

This homework assignment is a **Kaggle-style competition** where you will build a machine learning model to predict click-through rates (CTR) for display advertisements using the Criteo dataset. This assignment directly builds on concepts from the **Recommender Systems II** lecture, particularly the Deep Learning Recommender Model (DLRM) architecture.

**Duration:** 2.5 weeks from release date

**Type:** Individual or team-based competition (instructor discretion)

## Background and Motivation

Recommender systems and CTR prediction are critical components of modern digital advertising and content platforms:

- **Amazon**: Up to 35% of revenue attributed to recommendations
- **Netflix**: 75% of movies watched come from recommendations  
- **Meta/Facebook**: CTR prediction drives ad ranking and content recommendations
- **Industry Scale**: Over 79% of AI inference cycles in production data centers are dedicated to recommendation workloads

This assignment gives you hands-on experience with a real-world industrial dataset used for benchmarking state-of-the-art recommendation models.

## The Task

**Objective:** Predict whether a user will click on a display advertisement.

**Problem Type:** Binary classification with probabilistic outputs

**Dataset:** Criteo Ad Click-Through Rate Dataset

- **Source:** [Criteo Click Logs on HuggingFace](https://huggingface.co/datasets/criteo/CriteoClickLogs)
- **Original Challenge:** [Kaggle Criteo Display Ad Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge)

## Dataset Description

### Structure

The dataset contains feature values and click feedback for millions of display ads collected over 24 days of Criteo traffic.

- **Each row** represents one display ad served by Criteo
- **Target variable** (first column): Binary label
  - `1` = ad was clicked
  - `0` = ad was not clicked
- **Features:** 39 total features across two types
- **Format:** Tab-separated values (TSV)
- **Temporal ordering:** Rows are chronologically ordered

### Features

#### 13 Integer Features (Dense/Continuous)
- Represent numerical properties of the ad, user, or context
- Mostly count-based features
- Examples might include: time-based features, frequency counts, numerical user attributes
- **Note:** Semantic meanings are undisclosed for business confidentiality

#### 26 Categorical Features (Sparse)
- Represent categorical properties (user ID, item ID, location, device type, etc.)
- Values are **hashed into 32-bit integers** for anonymization
- High cardinality (some features have millions of unique values)
- **Note:** Semantic meanings are undisclosed for business confidentiality

### Important Notes

- **Missing Values:** Some features may contain missing values (empty fields)
- **Subsampling:** Both positive and negative examples have been subsampled at different rates
- **Anonymization:** All features are anonymized for privacy protection
- **Class Imbalance:** Typical CTR datasets have very low positive rates (often < 5%)

### Data Format

```
<label> <int_1> ... <int_13> <cat_1> ... <cat_26>
```

Example row:
```
0	1	1	5	0	1382	4	15	2	181	1	2		2	68fd1e64	80e26c9b	...
```

## Competition Structure

### Timeline

- **Week 1-2:** Model development and experimentation
- **Week 2.5:** Final submissions and report writing
- **Total Duration:** 2.5 weeks from assignment release

### Leaderboard

- **Public Leaderboard:** Calculated on 50% of test set (visible during competition)
- **Private Leaderboard:** Calculated on remaining 50% of test set (revealed after deadline)
- **Final Ranking:** Based on Private Leaderboard performance

### Submission Limits

- **Maximum:** 5 submissions per day
- **Format:** CSV file with predictions
- **Purpose:** Encourages thoughtful experimentation rather than overfitting to public leaderboard

### Submission File Format

Your submission file must be a CSV with exactly two columns:

```csv
id,prediction
0,0.023
1,0.891
2,0.156
...
```

- `id`: Row identifier for the test sample
- `prediction`: Predicted probability of click (value between 0 and 1)

## Evaluation Metrics

### Primary Metric: ROC AUC Score

**Area Under the Receiver Operating Characteristic Curve (ROC AUC)**

- Standard metric for CTR prediction in industry
- Measures the model's ability to rank positive examples higher than negative examples
- Range: 0.5 (random) to 1.0 (perfect)
- **Robust to class imbalance**

### Secondary Metric: Log Loss (Binary Cross-Entropy)

**Log Loss** measures the quality of probabilistic predictions:

$$\text{LogLoss} = -\frac{1}{N}\sum_{i=1}^{N} [y_i \log(p_i) + (1-y_i)\log(1-p_i)]$$

- Penalizes confident wrong predictions heavily
- Better for evaluating calibrated probabilities
- Will be reported but not used for ranking

## Technical Requirements

### Allowed Tools and Frameworks

You may use **any** tools, libraries, or frameworks you wish:

- **Deep Learning:** PyTorch, TensorFlow, Keras
- **Gradient Boosting:** XGBoost, LightGBM, CatBoost
- **Traditional ML:** scikit-learn (Logistic Regression, Random Forest, etc.)
- **Specialized RecSys libraries:** TensorFlow Recommenders, PyTorch-BigGraph

### Model Architecture Flexibility

You are free to implement any approach:

- **Deep Learning models:** DLRM (from lecture), DeepFM, Wide & Deep, Deep & Cross Network (DCN)
- **Gradient Boosted Trees:** Often very competitive for tabular data
- **Traditional ML:** Logistic regression, factorization machines
- **Ensemble methods:** Combine multiple models
- **Hybrid approaches:** Mix neural networks and tree-based methods

### Required Considerations

Your solution must:

1. **Handle missing values** appropriately (imputation, special encoding, or model-native handling)
2. **Process both feature types:** Dense (integer) and sparse (categorical) features
3. **Produce probability scores** between 0 and 1 (not just binary predictions)
4. **Be reproducible** with clear documentation

## Deliverables

### 1. Predictions File (`predictions.csv`)

- Upload to competition platform (details provided separately)
- Must follow exact format specification
- Contains probability predictions for all test samples

### 2. Technical Report (2-3 pages)

Submit a PDF report addressing the following:

#### a) Model Architecture and Rationale (30%)

- What model(s) did you choose and why?
- How does your approach connect to concepts from the DLRM lecture?
  - If using embeddings: what dimensions? Why?
  - If using feature interactions: how are they computed?
  - If using traditional ML: how do you handle categorical features?

#### b) Feature Engineering (25%)

- How did you handle missing values?
- Did you create any new features? (ratios, combinations, binning, etc.)
- How did you process categorical features? (embeddings, one-hot, target encoding, hashing)
- Did you normalize/scale the dense features?

#### c) Training Methodology (25%)

- Training/validation split strategy (random, temporal, stratified?)
- Hyperparameters: learning rate, batch size, regularization, tree depth, etc.
- Optimization algorithm and training schedule
- How did you prevent overfitting?
- Training time and computational resources used

#### d) Results and Analysis (20%)

- Final validation performance (ROC AUC and Log Loss)
- Public and private leaderboard positions
- What worked well? What didn't work?
- If you tried multiple approaches, compare them quantitatively
- Key insights learned from the dataset

## Grading Rubric

Total: 100 points

### Model Performance (40 points)

Performance on **private leaderboard**:

- **Top 25%:** 40 points
- **Top 50%:** 35 points  
- **Top 75%:** 30 points
- **Bottom 25%:** 25 points
- **Baseline not exceeded:** 20 points

A simple baseline model will be provided for reference.

### Technical Report (40 points)

- **Model Architecture & Rationale (12 pts):** Clear explanation with connections to lecture concepts
- **Feature Engineering (10 pts):** Thoughtful approach to preprocessing and feature creation
- **Training Methodology (10 pts):** Sound experimental design and hyperparameter choices
- **Results & Analysis (8 pts):** Insightful discussion of what worked and why

### Code Quality (20 points)

- **Reproducibility (10 pts):** Code runs successfully with clear setup instructions
- **Documentation (5 pts):** Comments, README, clear structure
- **Code organization (5 pts):** Modular, readable code following best practices

## Getting Started

### Step 1: Access the Dataset

The dataset is available on HuggingFace:

- **Main dataset:** [https://huggingface.co/datasets/criteo/CriteoClickLogs](https://huggingface.co/datasets/criteo/CriteoClickLogs)
- **Download:** You can download directly or use the HuggingFace `datasets` library

```python
from datasets import load_dataset
dataset = load_dataset("criteo/CriteoClickLogs")
```

**Note:** The full dataset is very large (>1TB). Start with a subset for rapid iteration.

### Step 2: Data Sampling Strategy

For faster iteration during development:

1. **Start small:** Use first 1M rows for initial experiments
2. **Validate approach:** Expand to 10M rows to verify scalability
3. **Final training:** Use as much data as computationally feasible

**Temporal splits recommended:** Since data is chronologically ordered, split by time rather than random sampling.

### Step 3: Build a Baseline

Start with a simple baseline to establish a performance floor:

**Option 1: Logistic Regression**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import FeatureHasher
```

**Option 2: Simple MLP**
```python
import torch.nn as nn
# 2-layer MLP with embeddings for categorical features
```

**Option 3: Gradient Boosting**
```python
import xgboost as xgb
# XGBoost often works well out-of-the-box for tabular data
```

### Step 4: Memory Management Tips

Categorical features can create memory issues:

- **Feature hashing:** Reduce dimensionality of categorical features
- **Embedding dimensions:** Start with 8-32 for categorical features
- **Batch processing:** Don't load entire dataset into memory
- **Data types:** Use `int32` instead of `int64` where possible

### Step 5: Validation Strategy

Recommended validation approaches:

1. **Temporal split:** Train on days 1-20, validate on days 21-23, test on day 24
2. **Hold-out set:** Keep 20% of data for validation
3. **Monitor both metrics:** Track both ROC AUC and Log Loss

## Bonus Challenges (Optional)

Extra credit opportunities (up to 10 bonus points):

### 1. Implement DLRM from Lecture (5 points)

- Implement the DLRM architecture as presented in the lecture
- Compare performance to your main approach
- Discuss the trade-offs (performance vs. complexity)

### 2. Multi-Model Comparison (3 points)

- Train at least 3 different model types (e.g., neural network, gradient boosting, linear model)
- Provide quantitative comparison on the same validation set
- Analyze which model types work best for this data

### 3. Feature Importance Analysis (2 points)

- Identify which features are most predictive
- Use techniques like SHAP values, permutation importance, or built-in feature importance
- Visualize and discuss findings

### 4. Handle Class Imbalance (2 points)

- Implement creative approaches to handle the imbalanced dataset
- Try techniques like: weighted loss, focal loss, oversampling, undersampling
- Measure impact on performance

### 5. Model Ensembling (3 points)

- Create an ensemble of multiple models
- Explore different ensemble strategies (averaging, stacking, boosting)
- Show improvement over individual models

## Resources

### From Course Materials

- **Lecture Slides:** [Recommender Systems II](./20-Recommender-Systems-II.qmd)
- **Key Concepts:** Embeddings, feature interactions, matrix factorization, DLRM architecture

### Research Papers

- **DLRM Paper:** Naumov et al. (2019). "Deep Learning Recommendation Model for Personalization and Recommendation Systems." [arXiv:1906.00091](https://arxiv.org/abs/1906.00091)
- **Recommended Reading:** Ricci et al. (2022). "Recommender Systems Handbook."

### Tutorials and Guides

- **PyTorch DLRM:** [Facebook Research DLRM Implementation](https://github.com/facebookresearch/dlrm)
- **Feature Engineering for CTR:** Many Kaggle kernels available from original competition
- **Handling Categorical Features:** Entity embeddings, target encoding, frequency encoding

### Computing Resources

- **Free Options:** Google Colab (with GPU), Kaggle Kernels (30 hrs/week GPU)
- **University Resources:** [Add specific resources available to your students]
- **Local Development:** CPU-only development is feasible with data subsampling

## Frequently Asked Questions

### Q: Do I need to implement DLRM?

**A:** No! Any approach is valid. DLRM is one option, but gradient boosting or simpler models may work just as well. Choose what you understand and can implement effectively.

### Q: The dataset is too large for my computer. What should I do?

**A:** Start with a subset (1-10M rows). You can still learn effectively and build competitive models with sampled data. Use stratified sampling to maintain class balance.

### Q: Can I use pre-trained models or code from GitHub?

**A:** Yes, you may use existing implementations as a starting point. However, you must:
1. Understand the code thoroughly
2. Adapt it appropriately for this dataset
3. Clearly cite sources in your report
4. Demonstrate your own contribution and understanding

### Q: How do I handle missing values?

**A:** Several strategies work:
- Fill with -1 or another sentinel value
- Use mean/median imputation for dense features
- Create a "missing" indicator feature
- Some models (XGBoost, CatBoost) handle missing values naturally

### Q: Should I use a time-based split or random split?

**A:** Time-based split is recommended because:
1. The data is temporally ordered
2. It better mimics real-world deployment (predict future from past)
3. Prevents data leakage from future information

### Q: Can I work in a team?

**A:** [Instructor to specify: individual vs. team-based]

### Q: What if my model doesn't improve over the baseline?

**A:** That's okay! The report is 40% of your grade. Focus on:
1. Demonstrating understanding of the techniques
2. Explaining what you tried and why
3. Analyzing results thoughtfully
4. Making clear connections to course concepts

## Submission Instructions

### Predictions File

- **Filename:** `predictions.csv`
- **Upload to:** [Competition platform URL to be provided]
- **Deadline:** [Date and time to be specified]
- **Format:** Strict adherence to CSV format required

### Technical Report

- **Filename:** `report.pdf` (or `report_[teamname].pdf`)
- **Upload to:** [Course management system]
- **Deadline:** [Same as predictions deadline]
- **Format:** PDF only, 2-3 pages (excluding figures)

### Code Submission

- **Filename:** `code.zip` or link to GitHub repository
- **Upload to:** [Course management system]  
- **Include:**
  - All training/inference code
  - Requirements.txt or environment.yml
  - README with setup and running instructions
  - Any preprocessing scripts

## Academic Integrity

- You may discuss general approaches with classmates
- You may use online resources and cite them appropriately
- **You may not:** Share code, trained models, or prediction files directly
- **You may not:** Use predictions from others' models
- All violations will be taken seriously per university policy

## Tips for Success

1. **Start early:** Don't underestimate data loading and preprocessing time
2. **Iterate quickly:** Begin with small data samples and simple models
3. **Monitor overfitting:** Track validation performance, not just training
4. **Read the lecture:** Concepts from DLRM provide valuable intuition
5. **Experiment systematically:** Change one thing at a time and track results
6. **Document as you go:** Don't wait until the end to write your report
7. **Check submissions carefully:** Ensure file format is correct before uploading
8. **Ask questions:** Use office hours and discussion forums

## Support and Help

- **Office Hours:** [Times to be specified]
- **Discussion Forum:** [Platform to be specified]
- **Email:** [Instructor/TA contact]
- **Technical Issues:** Report platform issues immediately

---

**Good luck, and may your gradients descend smoothly!**

*This assignment is designed to give you practical experience with industrial-scale recommendation systems while deepening your understanding of the theoretical concepts from lecture.*

