import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
data = pd.read_csv(r'D:\AI_Projects\Project_1\data\processed\diabetes_clean.csv', sep= ';')
x = data.drop("Outcome", axis=1)
y = data["Outcome"]
# Chuẩn hóa standardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

# Chuyển về dataframe
X_scaled = pd.DataFrame(X_scaled, columns=x.columns)

# Lưu scaler
joblib.dump(scaler, r'D:\AI_Projects\Project_1\models\scaler.pkl')