from flask import Flask, request, jsonify
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Definición de universos
stock = ctrl.Antecedent(np.arange(0, 101, 1), 'stock')
umbral = ctrl.Antecedent(np.arange(0, 101, 1), 'umbral')
riesgo = ctrl.Consequent(np.arange(0, 1.1, 0.01), 'riesgo')

# Funciones de pertenencia para STOCK
stock['muy_bajo'] = fuzz.sigmf(stock.universe, 15, -0.2)
stock['bajo'] = fuzz.gaussmf(stock.universe, 25, 10)
stock['medio'] = fuzz.gaussmf(stock.universe, 50, 12)
stock['alto'] = fuzz.gaussmf(stock.universe, 75, 10)
stock['muy_alto'] = fuzz.sigmf(stock.universe, 85, 0.2)

# Funciones de pertenencia para UMBRAL
umbral['critico'] = fuzz.sigmf(umbral.universe, 15, -0.2)
umbral['bajo'] = fuzz.gaussmf(umbral.universe, 30, 10)
umbral['medio'] = fuzz.gaussmf(umbral.universe, 60, 10)
umbral['alto'] = fuzz.sigmf(umbral.universe, 80, 0.2)

# Funciones de pertenencia para RIESGO
riesgo['nulo'] = fuzz.sigmf(riesgo.universe, 0.1, -10)
riesgo['bajo'] = fuzz.gaussmf(riesgo.universe, 0.2, 0.1)
riesgo['medio'] = fuzz.gaussmf(riesgo.universe, 0.5, 0.1)
riesgo['alto'] = fuzz.gaussmf(riesgo.universe, 0.75, 0.1)
riesgo['extremo'] = fuzz.sigmf(riesgo.universe, 0.78, 14)

# Reglas difusas 
regla1 = ctrl.Rule(stock['muy_bajo'] & umbral['critico'], riesgo['extremo'])
regla2 = ctrl.Rule(stock['muy_bajo'] & (umbral['medio'] | umbral['alto']), riesgo['alto'])

regla3 = ctrl.Rule(stock['bajo'] & umbral['critico'], riesgo['alto'])
regla4 = ctrl.Rule(stock['bajo'] & umbral['medio'], riesgo['medio'])

regla5 = ctrl.Rule(stock['medio'] & umbral['critico'], riesgo['alto'])
regla6 = ctrl.Rule(stock['medio'] & umbral['medio'], riesgo['medio'])
regla7 = ctrl.Rule(stock['medio'] & ~umbral['alto'], riesgo['medio'])  # NOT lógico

regla8 = ctrl.Rule(stock['alto'] & (umbral['alto'] | umbral['medio']), riesgo['bajo'])
regla9 = ctrl.Rule(stock['muy_alto'] & umbral['bajo'], riesgo['nulo'])

# Regla multivariable con comportamiento inesperado
regla10 = ctrl.Rule((stock['muy_bajo'] & umbral['critico']) | (stock['bajo'] & umbral['alto']), riesgo['extremo'])

# Regla inverso lógico
regla11 = ctrl.Rule(~stock['muy_alto'] & umbral['critico'], riesgo['alto'])

# Regla de excepción (evitar falsos negativos)
regla12 = ctrl.Rule(stock['muy_bajo'] & ~umbral['bajo'], riesgo['extremo'])

# Reglas reforzadas (simulan ponderación)

# Casos de bajo stock que podrían no activarse tan fuerte
regla13 = ctrl.Rule(stock['muy_bajo'] & ~umbral['bajo'], riesgo['extremo'])
regla14 = ctrl.Rule(stock['muy_bajo'] & umbral['medio'], riesgo['extremo'])
regla15 = ctrl.Rule(stock['bajo'] & umbral['critico'], riesgo['extremo'])  # duplicada y elevada

# Casos intermedios pero con tendencia a riesgo elevado
regla16 = ctrl.Rule(stock['medio'] & umbral['critico'], riesgo['alto'])  # ya existe, pero se refuerza por ambigüedad
regla17 = ctrl.Rule(stock['medio'] & ~umbral['bajo'], riesgo['alto'])

# Regla redundante de validación final
regla18 = ctrl.Rule((stock['muy_bajo'] | stock['bajo']) & (umbral['critico'] | umbral['medio']), riesgo['extremo'])


# Sistema de control
sistema_ctrl = ctrl.ControlSystem([regla1, regla2, regla3, regla4, regla5, regla6, regla7, regla8, 
    regla9, regla10, regla11, regla12, regla13, regla14, regla15, regla16, regla17, regla18
])
sistema = ctrl.ControlSystemSimulation(sistema_ctrl)

@app.route('/evaluar', methods=['POST'])
def evaluar_lote():
    productos = request.json
    resultados = []

    for item in productos:
        nombre = item.get('producto') or item.get('Producto') or 'sin_nombre'
        stock_val = item.get('stock', 0)
        umbral_val = item.get('umbral', 0)

        sistema.input['stock'] = stock_val
        sistema.input['umbral'] = umbral_val
        sistema.compute()
        riesgo_valor = sistema.output['riesgo']

       # Clasificación cualitativa mejorada
        if riesgo_valor >= 0.83:
            riesgo_cualitativo = "EXTREMO"
        elif riesgo_valor >= 0.7:
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
