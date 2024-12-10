
# Diabetes Prediction using Machine Learning and Deep Learning

This repository contains a Python-based application for predicting diabetes risk using various machine learning models and a deep learning model. The models are trained on a dataset that includes features like pregnancies, glucose levels, BMI, and more. The application also provides a Gradio interface for interactive predictions.

## Requirements

Ensure you have the following libraries installed:

```bash
pip install pandas numpy scikit-learn xgboost tensorflow gradio joblib
```

## Project Structure

- `diabetes_prediction.py`: The main script that contains the code for data processing, model training, evaluation, and Gradio interface.
- `models/`: Directory where trained models will be saved.
- `best_deep_learning_model.keras`: The best-performing deep learning model saved during training.

## Dataset

The dataset used for training models contains the following features:

- **Pregnancies**: The number of pregnancies.
- **Glucose**: Plasma glucose concentration.
- **BloodPressure**: Blood pressure value.
- **SkinThickness**: Thickness of the skin fold.
- **Insulin**: Insulin levels in the blood.
- **BMI**: Body mass index.
- **DiabetesPedigreeFunction**: Diabetes pedigree function score.
- **Age**: Age of the individual.
- **Outcome**: The target variable indicating whether the individual has diabetes (1) or not (0).

### Data:

Data Source : https://github.com/ShadiM0hamed/Diabetes_ML-DL/tree/main

## Models Used

This project uses multiple machine learning models:

1. **Logistic Regression**
2. **Support Vector Machine**
3. **Decision Tree**
4. **Random Forest**
5. **K-Nearest Neighbors**
6. **Naive Bayes**
7. **XGBoost**

Additionally, an **Advanced Deep Learning Model** is implemented using TensorFlow with a neural network architecture.

## Model Training and Evaluation

- The dataset is split into training and testing sets.
- Models are trained using the training set and evaluated based on accuracy on the testing set.
- The best model is selected based on the highest accuracy.
- The trained models are saved as `.pkl` files for future predictions.
- The best deep learning model is saved in `.keras` format.

## Deep Learning Model

The deep learning model is a neural network with the following architecture:

1. Input layer
2. Hidden layers with `ReLU` activation, batch normalization, and dropout for regularization.
3. Output layer with `sigmoid` activation for binary classification.

## Gradio Interface

Gradio is used to create an interactive web interface that allows users to input values for the features and get predictions on whether an individual is at risk of diabetes.

### Input Fields:

- Model selection (ML or Deep Learning)
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

### Output:

- The prediction of the diabetes risk (probability).

## How to Run

1. Clone the repository or download the script.
2. Install the necessary libraries using `pip install -r requirements.txt`.
3. Run the script.

This will launch the Gradio interface in your browser.

## Example Output

Once you select a model and provide the necessary input, the application will display the prediction:

```
Prediction: 0.85
```

This indicates a prediction probability of 85% that the individual has diabetes.

## Saving and Loading Models

Models are saved using `joblib` for machine learning models and `TensorFlow` for the deep learning model. You can reload these models to make predictions after the initial training.

## Conclusion

This project demonstrates a full workflow for building, evaluating, and deploying machine learning and deep learning models for diabetes prediction. It uses a variety of models, optimizes them, and integrates them into a user-friendly interface with Gradio for easy interaction.
