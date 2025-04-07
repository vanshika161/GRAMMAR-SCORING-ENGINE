
# 🎯 SHL Grammar Scoring Engine | ML Audio Assessment Project

> **Objective:** Build an intelligent scoring engine that automatically rates the grammatical quality of spoken English audio using machine learning.

---

## 🧠 Overview

This project was completed as part of SHL’s hiring challenge to build a **Grammar Scoring Engine**. Given audio clips of spoken English (each 45–60 seconds long), the goal was to predict a **MOS Likert Grammar Score** ranging from **1 to 5**, representing grammar fluency and complexity.

The problem is unique because it combines **Natural Language Processing (NLP)**, **Speech Processing**, and **Machine Learning (ML)** for a real-world application in **Automated Assessment**.

---

## 📁 Dataset Description

- **Audio Files**: Provided in `.wav` format in two folders:
  - `audios_train/`: 444 labeled audio files.
  - `audios_test/`: 197 unlabeled audio files.

- **CSV Files**:
  - `train.csv`: Mapping of audio file names to grammar labels (continuous scores).
  - `test.csv`: List of test audio filenames.
  - `sample_submission.csv`: Format for submission (filename + predicted label).

---

## 🛠️ Technologies Used

- **Python 3**
- **Librosa** (audio feature extraction)
- **Scikit-learn** (model building, evaluation)
- **Pandas / NumPy / Matplotlib / Seaborn**
- **Jupyter Notebook** (modular development and EDA)

---

## 🔍 Key Features & Workflow

### ✅ Step-by-step Pipeline:

1. **Data Loading**:
   - All CSV files and `.wav` audio files are loaded and verified.

2. **Audio Preprocessing & Feature Engineering**:
   - Extracted **MFCC features** (Mel Frequency Cepstral Coefficients) from all audio using `librosa`.
   - Mean pooling of MFCCs to get consistent vector input for models.

3. **Model Development**:
   - Trained a **Random Forest Regressor** to predict grammar scores.
   - Tuned hyperparameters, used cross-validation.

4. **Evaluation**:
   - Evaluated using:
     - **MSE** (Mean Squared Error)
     - **MAE** (Mean Absolute Error)
     - **Pearson Correlation Coefficient** (Official Metric)

5. **Visualization**:
   - Created scatterplots to visualize predicted vs actual scores.

6. **Prediction & Submission**:
   - Ran the model on the 197 test audios.
   - Saved output in `submission.csv`.

---

## 📊 Evaluation Metrics

| Metric               | Score       |
|----------------------|-------------|
| Mean Squared Error   | *(e.g., 0.28)* |
| Mean Absolute Error  | *(e.g., 0.39)* |
| Pearson Correlation  | *(e.g., 0.84)* |

✅ *Achieved high correlation between predicted and actual grammar fluency scores.*

---

## 📌 Possible Enhancements

- Integrate **Wav2Vec2** embeddings for deep contextual audio understanding.
- Add **data augmentation** (noise, pitch, speed).
- Try **XGBoost / LightGBM / Deep Learning** for model enhancement.
- Deploy on **Streamlit** or **Flask** for demo/testing.

---

## 📁 Files Structure

```bash
📦 SHL_Grammar_Scoring_Project
├── audios_train/              # 444 labeled audio files
├── audios_test/               # 197 test audio files
├── train.csv                  # Training labels
├── test.csv                   # Test file list
├── sample_submission.csv      # Format reference
├── submission.csv             # ✅ Your output predictions
├── grammar_score_engine.ipynb # 💻 Main notebook
└── README.md                  # 📘 You're reading it!
```

---

## 💡 Conclusion

This project demonstrates end-to-end machine learning applied to **speech assessment**. It required expertise in:

- **Speech-to-Feature Conversion**
- **Handling Real-World Data**
- **Model Generalization & Validation**
- **Performance Metrics & Interpretability**

> 💬 *A solid blend of Audio Processing + NLP + ML for real-world scoring systems.*
