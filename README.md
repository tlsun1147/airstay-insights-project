# Airstay Insights – Airbnb Price Prediction (SageMaker XGBoost)

## Project Overview

This project builds a machine learning model to predict Airbnb listing prices in New York City using historical data. The solution is implemented using Amazon SageMaker to demonstrate a full cloud-based machine learning pipeline, including data preprocessing, model training, deployment, evaluation, and cleanup.

## Business Problem

Airstay Insights is a real estate analytics company that helps investors identify profitable Airbnb properties. Predicting listing prices enables investors to estimate expected revenue, compare pricing across neighborhoods, identify high-performing property types, and make data-driven investment decisions.

## Dataset

Source: NYC Airbnb Open Data (Kaggle)
Size: Approximately 48,000 listings
Target Variable: price

Key Features:

* neighbourhood_group (borough)
* neighbourhood
* room_type
* minimum_nights
* number_of_reviews
* availability_365
* calculated_host_listings_count

## Data Access

The dataset can be downloaded from Kaggle:
https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data

## Machine Learning Pipeline

### 1. Data Preparation

* Removed duplicates and irrelevant columns (id, name, host_id, host_name, etc.)
* Handled missing values (reviews_per_month filled with 0)
* Filtered out extreme price outliers (price > 500)
* Applied one-hot encoding to categorical variables

### 2. Exploratory Data Analysis (EDA)

* Visualized price distribution to detect skewness and outliers
* Compared average prices across room types and boroughs
* Identified location and room type as key predictors

### 3. Model Training (Amazon SageMaker)

Algorithm: XGBoost (Regression)
Platform: Amazon SageMaker
Instance Type: ml.m5.large
Storage: Amazon S3

Training was performed using SageMaker’s built-in XGBoost container with defined hyperparameters and separate train and validation datasets.

### 4. Model Deployment

* Model deployed to a temporary SageMaker endpoint
* Predictions generated in batches to avoid payload size limits
* Endpoint deleted after use to minimize AWS costs

### 5. Model Evaluation

Model performance was evaluated on a test dataset:

* RMSE: approximately 60
* MAE: approximately 40
* R²: approximately 0.5

These results indicate the model provides a reasonable baseline for predicting Airbnb prices, with acceptable error given natural variability in listing prices.

## Key Insights

* Manhattan listings have the highest average prices
* Entire homes are significantly more expensive than shared rooms
* Location and room type are the strongest predictors of price

## Repository Structure

airstay-insights-project/
│
├── notebooks/
│   └── assignment_5_1_airstay_insights_model_training.ipynb
│
├── data/
│   └── AB_NYC_2019.csv
│
├── README.md

## Technologies Used

* Python (Pandas, NumPy, Matplotlib, Scikit-learn)
* Amazon SageMaker
* Amazon S3
* XGBoost

## How to Run the Project

1. Download the dataset from Kaggle and upload it to your SageMaker environment
2. Open the notebook:
   notebooks/assignment_5_1_airstay_insights_model_training.ipynb
3. Run all cells sequentially
4. Ensure AWS permissions are configured for SageMaker and S3
5. The model will train, deploy, evaluate, and clean up resources automatically

## Future Enhancements

* Hyperparameter tuning using SageMaker HyperParameterTuner
* Feature engineering such as seasonality and location clustering
* Ensemble models for improved accuracy
* Real-time pricing recommendation system

## Team

Paul Matta
Titus Sun

## Notes

This project demonstrates a complete end-to-end machine learning workflow in the cloud, from raw data ingestion to deployed model and evaluation, following best practices for scalability and cost management.
