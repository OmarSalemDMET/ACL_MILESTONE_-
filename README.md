# Airline Customer Satisfaction Analysis and Prediction

## Project Overview

This repository contains a data analysis project focused on airline passenger data. The primary goal is to clean, preprocess, and analyze multiple datasets to build a machine learning model capable of predicting customer satisfaction.

The analysis is performed in the **Milestone_1.ipynb** notebook and includes:

* Loading and cleaning four distinct airline-related datasets.
* Extensive data preprocessing, including handling missing values and encoding categorical features.
* Feature selection using **Recursive Feature Elimination (RFE)** to identify the most impactful variables.
* Building, training, and evaluating a **neural network model** to predict customer satisfaction scores.
* Explaining the model's predictions using **SHAP**.

---

## Datasets Used

This project analyzes four datasets to get a comprehensive view of customer experience:

1. **AirlineScrappedReview_Cleaned.csv** – Scraped customer reviews including review text, ratings, traveller type, and class.
2. **Passanger_booking_data.csv** – Passenger booking information (sales channel, trip type, purchase lead time, etc.).
3. **Customer_comment.csv** – Customer comments and sentiment analysis labels.
4. **Survey data_Inflight Satisfaction Score.csv** – Detailed inflight survey dataset; primary source for model building.

---

## Notebook: Milestone_1.ipynb

This Jupyter Notebook contains the complete workflow for the project.

### Key Steps

#### **1. Data Loading**

* Loads four CSV files using `pandas` and `kagglehub`.

#### **2. Data Cleaning & EDA**

* Identifies and handles missing values.
* Fills or drops null values using strategies such as:

  * Filling text fields with `'Unknown'` or `'neutral'`.
  * Filling numerical fields with the median.
* Visualizes relationships using **seaborn** and **matplotlib**.

#### **3. Data Preprocessing**

* Converts categorical features into numerical values using:

  * **One-Hot Encoding** (`pd.get_dummies`)
  * **LabelEncoder**
* Applies preprocessing steps across all four dataframes.

#### **4. Feature Selection**

* Uses **RandomForestClassifier + RFE** to identify important features.
* Outputs feature rankings for each dataset.

#### **5. Model Building (Neural Network)**

* Trains a neural network using **TensorFlow/Keras**.
* Targets the *satisfaction score* in the survey dataset.
* The target is one-hot encoded into categories.
* Data is scaled using `StandardScaler`.

#### **6. Model Training & Evaluation**

Evaluated using:

* Accuracy
* Precision, Recall, F1-Score
* Classification Report
* Confusion Matrix

#### **7. Inference & Explainability (XAI)**

* Includes `predict_satisfaction` function for new predictions.
* **SHAP** is used to explain feature importance and model behavior.

---

## How to Use

1. Install required libraries:

   ```bash
   pip install pandas seaborn matplotlib scikit-learn tensorflow kagglehub shap
   ```
2. Open the `Milestone_1.ipynb` notebook.
3. Run all cells to reproduce the full analysis.

---

## Files in this Repository

* **Milestone_1.ipynb** – Full analysis and model workflow.
* **AirlineScrappedReview_Cleaned.csv** – Scraped airline review dataset.
* **README.md** – Project documentation.

---

## Author

Omar Abdelhamid Yehia El-Adly
