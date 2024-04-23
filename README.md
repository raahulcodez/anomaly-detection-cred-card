# **DETECTING ANOMALIES IN FINANCIAL TRANSACTIONS:**

This project aims to optimize and fortify each phase of fraud detection, significantly improving efficiency and accuracy. It aims to proactively prevent fraudulent transactions with a reduced utilization of resources. 

## **Goal**

- Enhance fraud detection through improved feature engineering, incorporating transaction frequency and user behavior.
- Implement real-time monitoring with sub-second latency, providing prompt insights into transaction patterns.
- Utilize behavioral analytics to identify evolving patterns, contributing to the reduction in false positives.

## **Methodology**

- Three machine learning models are employed: Logistic Regression, Decision Tree, and XGBoost.
- Each model predicts anomalies independently based on the features engineered.
- The predictions of the three models are combined using a Majority Voting rule.
- The final prediction is determined by the majority vote of the three models, ensuring a more robust result.

## **How to Run the Project**

1. Clone the project repository by executing the following command in the terminal:
    
    ```
    git clone https://github.com/raahulcodez/anomaly-detection-cred-card
    ```
    
2. Navigate to the project directory.
3. Install the required dependencies by running:
    
    ```
    pip install -r requirements.txt
    ```
    
4. Finally, run the web application by executing:
    
    ```
    streamlit run app.py
    ```
