 = data.drop("Outcome", axis=1)
y = data["Outcome"]
# Chuẩn hóa standardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Chuyển về dataframe
x_scaled = pd.DataFrame(x_scaled, columns=x.column