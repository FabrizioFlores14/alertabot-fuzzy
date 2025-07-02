from flask import Flask, request, jsonify
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

app = Flask(__name__)

stock = ctrl.Antecedent(np.arange(0, 100, 1), 'stock')
umbral = ctrl.Antecedent(np.arange(0, 100, 1), 'umbral')
riesgo = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'riesgo')

stock['bajo'] = fuzz.trapmf(stock.universe, [0, 0, 10, 30])
stock['medio'] = fuzz.trimf(stock.universe, [20, 50, 80])
stock['alto'] = fuzz.trapmf(stock.universe, [60, 80, 100, 100])

umbral['bajo'] = fuzz.trapmf(umbral.universe, [0, 0, 10, 30])
umbral['medio'] = fuzz.trimf(umbral.universe, [20, 50, 80])
umbral['alto'] = fuzz.trapmf(umbral.universe, [60, 80, 100, 100])

riesgo['bajo'] = fuzz.trimf(riesgo.universe, [0, 0.2, 0.4])
riesgo['medio'] = fuzz.trimf(riesgo.universe, [0.3, 0.5, 0.7])
riesgo['alto'] = fuzz.trimf(riesgo.universe, [0.6, 0.8, 1.0])

rules = [
    ctrl.Rule(stock['bajo'] & umbral['alto'], riesgo['alto']),
    ctrl.Rule(stock['medio'] & umbral['medio'], riesgo['medio']),
    ctrl.Rule(stock['alto'] & umbral['bajo'], riesgo['bajo']),
    ctrl.Rule(stock['medio'] & umbral['alto'], riesgo['alto']),
    ctrl.Rule(stock['bajo'] & umbral['medio'], riesgo['alto']),
    ctrl.Rule(stock['alto'] & umbral['alto'], riesgo['medio']),
    ctrl.Rule(stock['alto'] & umbral['medio'], riesgo['medio']),
    ctrl.Rule(stock['bajo'] & umbral['bajo'], riesgo['medio']),
]

controlador = ctrl.ControlSystem(rules)
evaluador = ctrl.ControlSystemSimulation(controlador)

@app.route("/evaluar", methods=["POST"])
def evaluar():
    data = request.get_json()
    evaluador.input['stock'] = float(data['stock'])
    evaluador.input['umbral'] = float(data['umbral'])
    evaluador.compute()

    riesgo_val = evaluador.output['riesgo']
    riesgo_cualitativo = (
        "ALTO" if riesgo_val > 0.7 else
        "MEDIO" if riesgo_val > 0.4 else
        "BAJO"
    )

    return jsonify({
        "riesgo": round(riesgo_val, 2),
        "riesgo_cualitativo": riesgo_cualitativo
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
