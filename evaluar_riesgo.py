from flask import Flask, request, jsonify
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Definimos los universos
stock = ctrl.Antecedent(np.arange(0, 101, 1), 'stock')
umbral = ctrl.Antecedent(np.arange(0, 101, 1), 'umbral')
riesgo = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'riesgo')

# Funciones de pertenencia para el stock
stock['bajo'] = fuzz.trimf(stock.universe, [0, 0, 50])
stock['medio'] = fuzz.trimf(stock.universe, [25, 50, 75])
stock['alto'] = fuzz.trimf(stock.universe, [50, 100, 100])

# Funciones de pertenencia para el umbral
umbral['bajo'] = fuzz.trimf(umbral.universe, [0, 0, 50])
umbral['medio'] = fuzz.trimf(umbral.universe, [25, 50, 75])
umbral['alto'] = fuzz.trimf(umbral.universe, [50, 100, 100])

# Funciones de pertenencia para el riesgo
riesgo['bajo'] = fuzz.trimf(riesgo.universe, [0, 0, 0.5])
riesgo['medio'] = fuzz.trimf(riesgo.universe, [0.3, 0.5, 0.7])
riesgo['alto'] = fuzz.trimf(riesgo.universe, [0.5, 1, 1])

# Reglas difusas
regla1 = ctrl.Rule(stock['bajo'] & umbral['alto'], riesgo['alto'])
regla2 = ctrl.Rule(stock['bajo'] & umbral['medio'], riesgo['medio'])
regla3 = ctrl.Rule(stock['bajo'] & umbral['bajo'], riesgo['medio'])
regla4 = ctrl.Rule(stock['medio'] & umbral['alto'], riesgo['medio'])
regla5 = ctrl.Rule(stock['medio'] & umbral['medio'], riesgo['medio'])
regla6 = ctrl.Rule(stock['medio'] & umbral['bajo'], riesgo['bajo'])
regla7 = ctrl.Rule(stock['alto'], riesgo['bajo'])

# Sistema de control
sistema_ctrl = ctrl.ControlSystem([regla1, regla2, regla3, regla4, regla5, regla6, regla7])
sistema = ctrl.ControlSystemSimulation(sistema_ctrl)

@app.route('/evaluar', methods=['POST'])
def evaluar_lote():
    productos = request.json
    resultados = []

    for item in productos:
        nombre = item.get('producto', 'sin_nombre')
        stock_val = item.get('stock', 0)
        umbral_val = item.get('umbral', 0)

        sistema.input['stock'] = stock_val
        sistema.input['umbral'] = umbral_val
        sistema.compute()
        riesgo_valor = sistema.output['riesgo']

        # Clasificación cualitativa
        if riesgo_valor >= 0.7:
            riesgo_cualitativo = "ALTO"
        elif riesgo_valor >= 0.4:
            riesgo_cualitativo = "MEDIO"
        else:
            riesgo_cualitativo = "BAJO"

        resultados.append({
            "producto": nombre,
            "riesgo": round(riesgo_valor, 2),
            "riesgo_cualitativo": riesgo_cualitativo
        })

    return jsonify(resultados)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
