import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the stock data from a file into a DataFrame
df = pd.read_csv('intelStock.csv')

X = df[['Year', 'Year Open', 'Year High', 'Year Low', 'Year Close']]
y = df['Average Stock Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a linear regression model using the training data
lr = LinearRegression()
lr.fit(X_train, y_train)

# Evaluate the model's performance on the testing data
y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean squared error:', mse)
print('Coefficient of determination:', r2)

# Visualize the predicted vs. actual stock prices using Matplotlib
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Average Stock Price')
plt.ylabel('Predicted Average Stock Price')
plt.title('Linear Regression Model')
plt.show()
