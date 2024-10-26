import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Carregar o dataset a partir de um arquivo CSV
data = pd.read_csv('vendas_sorvete.csv')

# Separar as variáveis independentes (X) e a variável dependente (y)
X = data[['Temperatura (°C)']]  # Entrada
y = data['Vendas de Sorvete (unidades)']  # Saída

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o modelo de regressão linear
modelo = LinearRegression()

# Treinar o modelo
modelo.fit(X_train, y_train)

# Avaliar o modelo (opcional, mas útil para verificar a acurácia)
score = modelo.score(X_test, y_test)
print(f"R² do modelo: {score}")

# Salvar o modelo treinado em um arquivo .pkl
with open('modelo_regressao.pkl', 'wb') as f:
    pickle.dump(modelo, f)