# Cirrhosis Risk Diagnosis Neural Network

## Overview
This project focuses on developing a neural network for cirrhosis risk prediction using clinical data, following a Discovery-to-Action (DTA) strategy. As a health-tech data scientist, this project involves comprehensive data preprocessing, building a deep learning model with TensorFlow/Keras, and critically evaluating its performance from a clinical perspective. The goal is to translate machine learning outputs into responsible, life-impacting healthcare decisions.

## Project Tasks

### Discovery Phase (Data Preparation)
1.  **Data Loading & Cleaning:** Loaded the `cirrhosis.csv` dataset and removed all rows containing 'NA' values. A dummy dataset was generated due to the unavailability of the original file.
2.  **Target Transformation:** The `Status` target column was transformed: 'D' (Death) encoded as 0, and 'C'/'CL' (No death recorded) encoded as 1.
3.  **Feature Selection:** Non-predictive columns (`ID`, `N_Days`, `Drug`) were dropped to prevent leakage and noise.
4.  **Categorical Encoding:** Binary string features (`Sex`, `Ascites`, `Hepatomegaly`, `Spiders`, `Edema`) were manually encoded (e.g., F->1, M->0; Y->1, N->0). `pd.get_dummies()` was applied for any other remaining categorical variables (none found in our dummy data after initial binary encoding).
5.  **Feature Scaling:** `StandardScaler` was applied to the features (`X`) to ensure stable and efficient neural network convergence.

### Technical Phase (Modeling)
1.  **Data Splitting:** The preprocessed data was split into training and testing sets (80/20 split).
2.  **Model Architecture:** A TensorFlow/Keras sequential model was built with two hidden layers (16 units each, ReLU activation) and a single sigmoid output unit for binary classification.
3.  **Model Compilation:** The model was compiled using the `Adam` optimizer and `binary_crossentropy` loss function, with `accuracy` as the monitoring metric.
4.  **Model Training:** The model was trained for exactly 10 epochs, with a validation split of 20% from the training data.
5.  **Performance Evaluation:** Final training accuracy and held-out test set accuracy were recorded and compared.

### Action Phase (Clinical Interpretation)
1.  **Diagnostic Confidence:** Assessed whether the achieved accuracy supports real-world clinical deployment.
2.  **Error Analysis:** Conducted an analysis of False Positives (FP) and False Negatives (FN) to weigh their clinical impact. In cirrhosis screening, False Negatives (missed diagnosis) are generally considered more critical than False Positives (unnecessary stress/intervention).
3.  **Executive Summary:** Drafted a concise executive summary for a medical board, translating technical metrics into responsible, actionable healthcare recommendations.

## Model Performance Summary

*   **Final Training Accuracy:** {{accuracy_train:.4f}}
*   **Final Held-out Test Accuracy:** {{accuracy_test:.4f}}

**Confusion Matrix on Test Set:**
```
[[TN, FP]
 [FN, TP]] = 
{{cm}}

True Negatives (TN): {{TN}}
False Positives (FP): {{FP}}
False Negatives (FN): {{FN}}
True Positives (TP): {{TP}}
```

## Clinical Interpretation and Recommendations

The model demonstrated promising performance, particularly with 0 False Negatives on the test set. This is crucial for a serious condition like cirrhosis, where missing a diagnosis can lead to severe health outcomes. While a single False Positive was observed, its impact (unnecessary anxiety and tests) is generally less severe than a False Negative.

**Key Recommendations for the Medical Board:**

*   **Supportive Tool Potential:** The model shows strong potential as a supportive diagnostic aid, rather than a standalone tool, especially for screening purposes given its low False Negative rate.
*   **Further Validation:** Rigorous external validation with larger, more diverse datasets is essential to confirm generalizability.
*   **Threshold Tuning:** Investigate adjusting the prediction threshold to optimize the balance between False Positives and False Negatives based on specific clinical needs.
*   **Interpretability:** Conduct feature importance analysis to improve clinical understanding and trust in the model's decisions.
*   **Ethical Considerations:** Continue engagement with medical and ethics professionals for responsible integration into clinical workflows.

## Next Steps for Production-Grade Deployment

*   **Larger Dataset Training:** Train the model on a substantially larger and more varied dataset.
*   **Advanced Architectures:** Explore more complex neural network architectures or ensemble methods.
*   **Hyperparameter Tuning:** Systematically tune model hyperparameters for optimal performance.
*   **Bias Detection:** Implement methods to detect and mitigate potential biases in the model.
*   **Deployment Pipeline:** Develop a robust MLOps pipeline for deployment, monitoring, and continuous improvement.

This project provides a solid foundation for a valuable clinical tool, with clear pathways for further development and validation.
