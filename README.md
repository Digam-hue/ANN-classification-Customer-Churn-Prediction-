# 🏦 Customer Churn Prediction using Artificial Neural Networks

This project is a web application designed to predict customer churn for a bank. By inputting customer details such as credit score, age, and tenure, the model, built with an Artificial Neural Network (ANN), predicts the likelihood of a customer leaving the bank.

This tool can help banks proactively identify at-risk customers and implement retention strategies to improve customer loyalty and reduce financial loss.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://predict-my-churn.streamlit.app/)

---

## 🚀 Live Demo

You can interact with the live application here:

**[https://predict-my-churn.streamlit.app/](https://predict-my-churn.streamlit.app/)**

---

## 🔧 Tech Stack & Libraries

This project was built using the following technologies:

*   **Programming Language:** Python 3.10
*   **Machine Learning:** TensorFlow (Keras) for building the ANN model.
*   **Data Manipulation:** Pandas & NumPy
*   **Data Preprocessing:** Scikit-learn (StandardScaler, OneHotEncoder, LabelEncoder)
*   **Web Framework:** Streamlit for creating the interactive web interface.

---

## 📂 Project Structure & File Explanation

Here is a breakdown of the key files in this repository and their purpose:

| File Name                   | Description                                                                                                                                                            |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `app.py`                    | The main file that runs the Streamlit web application. It contains the user interface code, loads the pre-trained model and preprocessors, and handles the prediction logic. |
| `predict.ipynb`             | A Jupyter Notebook containing the end-to-end machine learning workflow. This includes data loading, preprocessing, model training, and the logic for the final prediction pipeline. |
| `model.h5`                  | The saved and trained Artificial Neural Network (ANN) model file created using Keras. This is the "brain" of the prediction.                                              |
| `scaler.pkl`                | A pickled `StandardScaler` object from Scikit-learn. It is used to scale new user input to the same distribution as the training data, which is crucial for model performance. |
| `onehot_encoder_geo.pkl`    | A pickled `OneHotEncoder` object. This is used to transform the categorical 'Geography' feature (e.g., 'France', 'Germany') into a numerical format that the model can understand. |
| `label_encoder_gender.pkl`  | A pickled `LabelEncoder` object used to convert the binary 'Gender' feature ('Male'/'Female') into numerical values (1/0).                                                |
| `requirements.txt`          | A text file listing all the Python libraries and dependencies required to run this project. It allows for easy replication of the environment.                           |

---

## 📈 The Machine Learning Pipeline

The prediction process follows these steps:

1.  **User Input:** The user provides 10 features about a customer through the web interface.
2.  **Data Structuring:** The inputs are collected into a Pandas DataFrame.
3.  **Categorical Encoding:**
    *   The `Gender` column is converted from text ('Male'/'Female') to numbers (1/0) using the loaded `LabelEncoder`.
    *   The `Geography` column is converted into separate binary columns (e.g., `Geography_France`, `Geography_Germany`) using the loaded `OneHotEncoder`.
4.  **Feature Scaling:** All numerical features are scaled using the loaded `StandardScaler` to ensure that no single feature dominates the model's learning process.
5.  **Prediction:** The fully preprocessed and scaled data is fed into the pre-trained ANN model (`model.h5`), which outputs a churn probability.
6.  **Display Result:** The application interprets the probability (>0.5 means likely to churn) and displays a user-friendly result.

---

## 🛠️ How to Run This Project Locally

To run this application on your own machine, follow these steps:

**Prerequisites:**
*   Python 3.8 - 3.11
*   Git

**Steps:**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Digam-hue/ANN-classification-Customer-Churn-Prediction-.git
    ```

2.  **Navigate to the project directory:**
    ```bash
    cd ANN-classification-Customer-Churn-Prediction-
    ```

3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

4.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

6.  Open your web browser and go to `http://localhost:8501`.

---

## 👨‍💻 Connect with Me

Feel free to connect with me for any questions or collaborations!

*   **GitHub:** [Digam-hue](https://github.com/Digam-hue)
*   **LinkedIn:** [Digambar Baditya](https://www.linkedin.com/in/digambar-baditya-b522b12a5/)
