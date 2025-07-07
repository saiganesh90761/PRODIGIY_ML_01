import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

train_data=pd.read_csv("train.csv")

print(train_data.head())

features = ['GrLivArea', 'FullBath', 'HalfBath','BedroomAbvGr']
x = train_data[features]
y=train_data["SalePrice"]
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.2,random_state=45)

model=LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label='Predicted vs Actual')

max_val = max(max(y_test), max(y_pred))
min_val = min(min(y_test), min(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction Line')

plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted Sale Price")
plt.legend()
plt.grid(True)
plt.show()
example_house = pd.DataFrame({
    'GrLivArea': [1710],
    'FullBath': [2],
    'HalfBath': [1],
    'BedroomAbvGr': [3]
})

predicted_price = model.predict(example_house)

print(f"Predicted Sale Price: ${predicted_price[0]:,.2f}")
