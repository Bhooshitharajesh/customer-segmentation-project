Customer Segmentation & Product Recommendation System

Project Overview

This project is a Machine Learning based Customer Segmentation and Product Recommendation System developed using Python, Streamlit, FastAPI, and FP-Growth Algorithm.

The application analyzes customer purchasing behavior using RFM (Recency, Frequency, Monetary) analysis and groups customers into meaningful segments using KMeans Clustering. Based on customer purchase patterns, the system also provides product recommendations using the FP-Growth association rule mining algorithm.

The project includes:

Customer Segmentation using KMeans Clustering ,
Product Recommendation System using FP-Growth ,
Interactive Streamlit UI ,
FastAPI deployment for customer segmentation prediction ,
Swagger UI testing interface ,

Features :

Customer Segmentation using RFM Analysis ,
KMeans Clustering based prediction ,
Product Recommendation System ,
FP-Growth Algorithm for association rule mining, 
Interactive Streamlit Dashboard ,
FastAPI integration for segmentation prediction ,
Swagger UI API testing ,
CSV Upload Support ,
Single Customer Prediction ,
Cluster Visualization and Analysis

Tech Stack :

Python ,
Pandas ,
NumPy ,
Scikit-learn ,
Streamlit ,
FastAPI ,
Matplotlib ,
MLxtend ,
Joblib

Machine Learning Workflow :

Customer Segmentation:

Data Cleaning and Preprocessing ,
RFM Feature Engineering ,
Feature Scaling ,
KMeans Clustering ,
Cluster Prediction ,
Segment Analysis and Visualization 

Customer Segments :

* Premium Customers
* Regular Customers
* Low Engagement Customers
  
Product Recommendation System :

The recommendation system is built using the FP-Growth Algorithm.

Workflow :

1. Transaction Data Preparation
2. Basket Matrix Creation
3. Frequent Itemset Generation using FP-Growth
4. Association Rule Mining
5. Product Recommendation Generation

The recommendation feature is integrated directly within the Streamlit UI.

FastAPI Integration

FastAPI is implemented for Customer Segmentation prediction.

The API allows:

Sending customer RFM inputs ,
Predicting customer segment ,
Testing through Swagger UI

Note:
The Product Recommendation feature is implemented inside the Streamlit application and not exposed through FastAPI.

Live Demo
Streamlit App :https://customer-segmentation-project-q9ttqvdahgdyauh4xuiuct.streamlit.app/
