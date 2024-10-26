from flask import Flask, request, render_template
import numpy as np
import pickle

# Criação da aplicação Flask
app = Flask(__name__)

# Carregar o modelo treinado
with open('modelo_regressao.pkl', 'rb') as f:
    modelo = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obter o valor de temperatura do formulário
        temperatura_vendas_sorvete = float(request.form['temperatura_vendas_sorvete'])

        # Transformar a entrada em um array adequado para o modelo
        temperatura_vendas_sorvete_array = np.array([[temperatura_vendas_sorvete]])

        # Fazer a previsão usando o modelo
        produtividade_prevista = modelo.predict(temperatura_vendas_sorvete_array)

        # Retornar o resultado em HTML
        return render_template('resultado.html', horas=temperatura_vendas_sorvete, produtividade=produtividade_prevista[0])
    
    except ValueError:
        # Em caso de erro na conversão de valor, retornar uma mensagem amigável
        return render_template('index.html', mensagem_erro="Por favor, insira um valor numérico válido.")

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Altere a porta conforme necessário
