# Multi-Modal Machine Learning for Early Trust Prediction in Human-AI Interaction by Biosignals

This repository contains the Python implementation used for the paper **"Multi-Modal Machine Learning for Early Trust Prediction in Human-AI Interaction Using Face Image and GSR Biosignals".  
The code demonstrates the complete pipeline for feature extraction, unimodal model training, and multi-modal stacking ensemble for trust prediction using biosignal and facial features.

---

## 🧠 Overview

The workflow includes:
1. Feature Extraction**
   - Face-based features from video segments using MediaPipe FaceMesh and ViT-Face-Expression.
   - GSR signal preprocessing and feature extraction using NeuroKit2.
2. Machine Learning Pipeline**
   - Data preprocessing, feature scaling, and resampling (SMOTETomek).
   - Unimodal model training (e.g., Random Forest, SVM, XGBoost).
   - Multi-modal stacking ensemble with logistic regression meta-learner.
3. Evaluation**
   - Accuracy, ROC-AUC, Brier Score, Precision, Recall, and F1 metrics.

---

## 🧩 File Description

| `From_feature_extraction_to_Multi_modal_Applied_Ergo.ipynb` | End-to-end notebook implementing multimodal feature extraction and stacking ML framework. |

---

## ⚙️ Requirements

Install the dependencies using pip:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, the main dependencies are:
```bash
pip install mediapipe
pip install torch torchvision torchaudio
pip install transformers
pip install scikit-learn
pip install imbalanced-learn
pip install numpy pandas matplotlib
pip install neurokit2
```

---

## ▶️ Usage

1. Open the notebook in Google Colab or Jupyter Notebook.
2. Upload the required data:
   - Zipped folder of face video segments.
   - Zipped folder and corresponding CSV file for GSR data.
3. Run all cells sequentially.
4. The notebook will:
   - Extract features from both modalities.
   - Train unimodal and multimodal models.
   - Save the performance metrics and plots in the output folder.

---

## 📊 Outputs

The notebook generates:
- **Feature CSV files** for face and GSR modalities.
- **Model performance results** in tabular and graphical formats.
- **Trained model objects** for reproducibility.

---

## 📚 Citation

If you use this code in your research, please cite as follows:

**IEEE format:**
> H. Shamszare, *Multi-Modal Machine Learning for Early Trust Prediction in Human-AI Interaction by Biosignals Code*, GitHub repository, 2025. [Online]. Available: https://github.com/hamid-shams/Multi-Modal-Machine-Learning-for-Early-Trust-Prediction-in-Human-AI-Interaction-by-Biosignals-Code-
**BibTeX:**
```bibtex
@misc{shamszare_multimodaltrust_2025,
  author       = {Hamid Shamszare},
  title        = {Multi-Modal Machine Learning for Early Trust Prediction in Human-AI Interaction by Biosignals Code},
  year         = {2025},
  howpublished = {\url{https://github.com/hamid-shams/Multi-Modal-Machine-Learning-for-Early-Trust-Prediction-in-Human-AI-Interaction-by-Biosignals-Code}},
  note         = {Accessed: Oct. 20, 2025}
}
```

---

## 👨‍💻 Author

**Hamid Shamszare**  
Ph.D. Candidate, Industrial & Management Systems Engineering  
West Virginia University, USA  
📧 Email: hs00055@mix.wvu.edu  
📘 [Google Scholar](https://scholar.google.com/citations?user=17Iz2mgAAAAJ&hl=en) 

---

## 📝 License


You are free to use, modify, and distribute the code with appropriate citation.

---

## 🌐 Future DOI

A permanent DOI will be assigned via [Zenodo](https://zenodo.org) after the official release of version 1.0.
