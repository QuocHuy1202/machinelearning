import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Tải và nạp dữ liệu
data = pd.read_csv("laptop_price.csv", encoding="latin1")
print("Dữ liệu ban đầu:")
print(data.head())

# 2. Thống kê mô tả dữ liệu
print("\nThông tin dữ liệu:")
print(data.info())
print("\nThống kê mô tả:")
print(data.describe())

# 3. Làm sạch dữ liệu
# Loại bỏ giá trị khuyết
print("\nSố lượng giá trị thiếu:")
print(data.isnull().sum())
data = data.dropna()

# Mã hóa dữ liệu dạng chuỗi
encoder = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = encoder.fit_transform(data[col])

# Phân tích tương quan
plt.figure(figsize=(12, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Ma trận tương quan")
plt.show()

# 4. Lựa chọn đặc trưng
corr_threshold = 0.3  # Ngưỡng chọn các đặc trưng có tương quan đáng kể
corr_matrix = data.corr()
target_corr = corr_matrix["Price_euros"]
selected_features = target_corr[abs(target_corr) > corr_threshold].index.tolist()
selected_features.remove("Price_euros")
print("\nCác đặc trưng được chọn:", selected_features)

# 5. Huấn luyện mô hình
X = data[selected_features]
y = data["Price_euros"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Huấn luyện nhiều mô hình để so sánh
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Support Vector Regressor (SVR)": SVR(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\nĐánh giá mô hình {name}:")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2 Score:", r2_score(y_test, y_pred))
