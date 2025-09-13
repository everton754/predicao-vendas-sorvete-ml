# 🍦 Predição de Vendas de Sorvete com Machine Learning

## 📋 Descrição do Projeto
Este repositório contém um projeto completo de Machine Learning para prever vendas diárias de sorvete com base na temperatura ambiente. O objetivo é auxiliar proprietários de sorveterias a otimizar a produção, reduzindo desperdícios e maximizando lucros.

## 🎯 Objetivos
- Treinar um modelo de regressão linear para prever vendas de sorvete.
- Registrar e gerenciar experimentos com MLflow.
- Implementar pipeline estruturado para garantir reprodutibilidade.
- Preparar deploy do modelo em ambiente de cloud (Azure Machine Learning).

## 📁 Estrutura do Repositório
```
predicao-vendas-sorvete-ml/
├── inputs/
│   ├── dados_sorvete_temperatura.csv  # Dataset com 365 observações
│   └── descricao.txt                  # Descrição do dataset
├── criar_dados.py                     # Script para gerar dados sintéticos
├── treinar_modelo.py                  # Script de treinamento sem tracking
├── treinar_mlflow.py                  # Script de treinamento com MLflow
├── .gitignore                         # Para ignorar venv, mlruns e arquivos pesados
├── LICENSE                            # Licença MIT
└── README.md                          # Documentação do projeto
```

## 📊 Dataset
- **Temperatura**: Temperatura média diária (°C).  
- **Vendas_Sorvete**: Unidades vendidas diárias.  
- **Observações**: 365 registros (um ano completo).  
- **Correlação**: 0.978 entre temperatura e vendas, indicando forte relação linear.

## 🔧 Tecnologias Utilizadas
- Python 3.8+  
- Pandas, NumPy para manipulação de dados  
- scikit-learn para regressão linear  
- MLflow para tracking e registro de modelos  
- Azure Machine Learning para deploy em cloud  

## 🚀 Como Usar

1. Clone o repositório:
   ```
   git clone https://github.com/everton754/predicao-vendas-sorvete-ml.git
   cd predicao-vendas-sorvete-ml
   ```

2. Crie e ative o ambiente virtual:
   ```
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Instale as dependências:
   ```
   pip install pandas numpy scikit-learn mlflow joblib jupyter
   ```

4. Gere o dataset sintético:
   ```
   python criar_dados.py
   ```

5. Treine o modelo localmente:
   ```
   python treinar_modelo.py
   ```

6. Faça tracking com MLflow:
   - Em uma aba do terminal:
     ```
     mlflow ui
     ```
   - Em outra aba:
     ```
     python treinar_mlflow.py
     ```
   - Acesse http://localhost:5000 para visualizar experimentos.

## 📈 Resultados do Modelo
- **Equação**: Vendas = 25.150 × Temperatura – 72.921  
- **R² Score**: 0.955 (média entre treino/teste)  
- **RMSE**: ~31 sorvetes  
- **MAE**: ~25 sorvetes  

## ☁️ Deploy em Produção
Use o Azure Machine Learning para registrar e implantar o modelo como um web service. Veja o script de exemplo em `deploy/azure_deployment.py`.

## 📝 Descrição do Dataset
O arquivo `inputs/descricao.txt` contém detalhes sobre as colunas e a origem sintética dos dados.

## 📜 Licença
Este projeto está licenciado sob a **MIT License**.
