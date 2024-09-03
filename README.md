# K-water Changwon Project: Turbidity and Coagulant Injection Rate Prediction
2023.09.04. ~ 2023.10.13.

This repository contains the implementation details of the project developed for **Korea Water Resources Corporation (K-water) Changwon Branch**.  
The primary focus of this project was to develop predictive models for:
- **Optimal Coagulant Injection Rate**
- **Turbidity Prediction**

## Project Overview

To demonstrate the practical application and deployment of the models, this repository includes an MLOps pipeline using various tools and technologies such as:

- **Docker**: Used for consistent environment setup across different stages of deployment, ensuring that the application runs reliably on any system.
- **Kafka**: Selected for its ability to handle and process large volumes of real-time data, which is critical for making timely predictions based on current water quality measurements.
- **MLflow**: Integrated for effective management of the machine learning lifecycle, including experiment tracking, model versioning, and streamlined deployment.
- **MySQL**: Employed as a robust database solution to store model predictions and performance metrics, providing a reliable backend for data-driven insights.
- **Grafana**: Chosen to visualize and monitor the system's performance and the model's predictions in real-time, allowing for proactive management.


<img width="567" alt="스크린샷 2024-09-03 오후 11 35 38" src="https://github.com/user-attachments/assets/0c48ae1d-849b-497b-9f3b-963d9f7d6e28">



## Confidentiality Notice

Please note that the datasets and trained models used for this project are confidential and cannot be shared publicly. All sensitive data and model files have been excluded from this repository.


## Reference
https://mlops-for-mle.github.io/tutorial/docs/intro
