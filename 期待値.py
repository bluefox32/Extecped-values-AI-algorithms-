import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# データの生成
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 線形回帰モデルの作成
model = LinearRegression()

# モデルの訓練
model.fit(X_train, y_train)

# 予測
y_pred = model.predict(X_test)

# 評価
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# 期待値の計算
expected_value = np.mean(y_pred)
print(f"Expected Value: {expected_value}")

# 誤差の分布
errors = y_test - y_pred
plt.hist(errors, bins=20)
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Distribution of Prediction Errors")
plt.show()

# 確率変数（予測値）
prob_variables = y_pred
print(f"Probability Variables (Predictions): {prob_variables[:5]}")

# 平均値の計算
actual_mean = np.mean(y_test)
predicted_mean = np.mean(y_pred)
print(f"Actual Mean: {actual_mean}")
print(f"Predicted Mean: {predicted_mean}")

# 訓練データと予測ラインのプロット
plt.scatter(X, y, color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression")
plt.show()