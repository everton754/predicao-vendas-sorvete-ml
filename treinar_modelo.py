import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. Carregar dados
data = pd.read_csv('dados_sorvete_temperatura.csv')

# 2. Preparar X e y
X = data[['Temperatura']]
y = data['Vendas_Sorvete']

# 3. Dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Treinar modelo
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Fazer predições
y_pred = model.predict(X_test)

# 6. Calcular métricas
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("✅ Modelo treinado com sucesso!")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.3f}")
print(f"Equação: Vendas = {model.coef_[0]:.3f} × Temperatura + {model.intercept_:.3f}")