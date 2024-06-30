import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import joblib

# データの生成
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特徴量の生成（多項式特徴量）
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 線形回帰モデルの作成
model = LinearRegression()

# モデルの訓練
model.fit(X_train_poly, y_train)

# 学習済みモデルの保存
joblib_file = "linear_regression_model_with_features.pkl"
joblib.dump(model, joblib_file)

# 予測
y_pred = model.predict(X_test_poly)

# 実際のデータを予測データに置き換え
y_test_replaced = y_test.copy()
replacement_indices = np.random.choice(len(y_test), size=len(y_test)//2, replace=False)
y_test_replaced[replacement_indices] = y_pred[replacement_indices]

# 評価（置き換え後）
mse_replaced = mean_squared_error(y_test, y_pred)
r2_replaced = r2_score(y_test, y_pred)
print(f"Mean Squared Error (Replaced): {mse_replaced}")
print(f"R^2 Score (Replaced): {r2_replaced}")

# 期待値の計算（置き換え後）
expected_value_replaced = np.mean(y_test_replaced)
print(f"Expected Value (Replaced): {expected_value_replaced}")

# 誤差の分布（置き換え後）
errors_replaced = y_test_replaced - y_pred
plt.hist(errors_replaced, bins=20)
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Distribution of Prediction Errors (Replaced)")
plt.show()

# 確率変数（予測値）
prob_variables_replaced = y_pred
print(f"Probability Variables (Predictions): {prob_variables_replaced[:5]}")

# 平均値の計算（置き換え後）
actual_mean_replaced = np.mean(y_test_replaced)
predicted_mean_replaced = np.mean(y_pred)
print(f"Actual Mean (Replaced): {actual_mean_replaced}")
print(f"Predicted Mean (Replaced): {predicted_mean_replaced}")

# 訓練データと予測ラインのプロット
plt.scatter(X, y, color='blue')
X_plot = np.linspace(0, 2, 100).reshape(-1, 1)
X_plot_poly = poly.transform(X_plot)
y_plot = model.predict(X_plot_poly)
plt.plot(X_plot, y_plot, color='red', linewidth=2)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression with Polynomial Features (Replaced)")
plt.show()

# 新しいデータに対する予測関数
def predict_new_data(new_data):
    """
    新しいデータに対して予測を行う関数
    :param new_data: 新しいデータ (numpy array)
    :return: 予測値
    """
    new_data_poly = poly.transform(new_data)
    prediction = model.predict(new_data_poly)
    return prediction

# 例として新しいデータポイントを生成
new_data_point = np.array([[1.5]])

# 新しいデータポイントに対する予測
prediction = predict_new_data(new_data_point)
print(f"Prediction for new data point {new_data_point[0][0]}: {prediction[0][0]}")