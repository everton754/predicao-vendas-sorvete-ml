import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Configurar experimento
mlflow.set_experiment("Predicao_Vendas_Sorvete")

with mlflow.start_run():
    # Carregar dados
    data = pd.read_csv('dados_sorvete_temperatura.csv')
    X = data[['Temperatura']]
    y = data['Vendas_Sorvete']

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Treinar modelo
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predições
    y_pred = model.predict(X_test)

    # Métricas
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Log parâmetros e métricas
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)


    # Log do modelo
    mlflow.sklearn.log_model(model, "model")

    print("✅ Run MLflow concluído!")
    print(f"RMSE: {rmse:.2f} | R2: {r2:.3f}")