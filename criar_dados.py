import numpy as np
import pandas as pd

# Gerar dados sintéticos
np.random.seed(42)
n_samples = 365

temperatura = np.random.uniform(15, 35, n_samples)
vendas = 50 + 15 * temperatura + np.random.normal(0, 30, n_samples)
vendas = np.maximum(vendas, 50)

data = pd.DataFrame({
    'Temperatura': temperatura,
    'Vendas_Sorvete': vendas
})

data.to_csv('dados_sorvete_temperatura.csv', index=False)
print("✅ Arquivo de dados criado com sucesso!")
print(f"📊 {len(data)} observações geradas")
print("\n🔍 Primeiras 5 linhas:")
print(data.head())