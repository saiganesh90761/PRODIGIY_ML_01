# 🏡 House Price Prediction using Linear Regression

This project demonstrates a simple machine learning model that predicts house sale prices using a **Linear Regression** algorithm from the **scikit-learn** library. It utilizes basic home features such as living area size, number of bathrooms, and bedrooms to estimate the price.

---

## 📁 Project Structure

project/
├── train.csv # Training dataset
├── house_price_predictor.py # Main Python script
└── README.md # This file



---

## 🧾 Dataset

- **File**: `train.csv`  
- Sourced from the **Kaggle House Prices** dataset.
- Key columns used for prediction:
  - `GrLivArea`: Above grade (ground) living area in square feet
  - `FullBath`: Number of full bathrooms
  - `HalfBath`: Number of half bathrooms
  - `BedroomAbvGr`: Number of bedrooms above ground
  - `SalePrice`: The target variable (price of the house)

---

## 🔍 Features Used for Training


features = ['GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr']
These features were selected for simplicity and are sufficient to demonstrate the regression model.

🧠 Model
Model Type: Linear Regression

Library: scikit-learn

Train-Test Split: 20% training, 80% testing

Metrics Used:

R² Score

Mean Squared Error (MSE)

Mean Absolute Error (MAE)

📊 Evaluation Output
Upon running the script, the following will be printed:

R² Score – how well the model explains the variance

MSE & MAE – error values to understand prediction accuracy

A scatter plot showing actual vs. predicted sale prices

💡 Example Prediction
The model also makes a prediction on a sample house:


example_house = pd.DataFrame({
    'GrLivArea': [1710],
    'FullBath': [2],
    'HalfBath': [1],
    'BedroomAbvGr': [3]
})
Output:
Predicted Sale Price: $200,000.00
(Example output may vary depending on data and model performance)

📦 Installation & Running
1. Install Dependencies

pip install pandas numpy matplotlib seaborn scikit-learn
2. Place train.csv in the same directory as the script.
3. Run the Script

python house_price_predictor.py
📚 References
Dataset: Kaggle - House Prices: Advanced Regression Techniques

scikit-learn: https://scikit-learn.org/

🏷️ Tags
Linear Regression House Prices Machine Learning Supervised Learning Python scikit-learn Data Science









Ask ChatGPT
