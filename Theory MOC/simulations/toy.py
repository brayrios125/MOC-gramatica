import numpy as np
from sklearn.cluster import KMeans
import random
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning
import networkx as nx  # ¡Falta esta línea!
# ------------------- PARÁMETROS GLOBALES -------------------
N_NEURONAS   = 50
CONECTIVIDAD = 0.1
TMAX         = 2000
LR_VAL       = 0.1
ETA_INIT     = 0.05
MU_META      = 0.01
ALPHA_SARSA  = 0.1
EPSILON      = 0.1
GAMMA        = 0.9
LAMBDA       = 0.8

# ------------------- CLASE 1: Semántica (Versión Robusta) -------------------
class Semantica:
    def __init__(self, max_cat=8):
        self.max_cat = max_cat
        self.data = []
        self.k = 2
        self.model = None
        self.pattern_len = -1

    def actualizar(self, patron):
        if not patron: return

        if self.pattern_len == -1:
            self.pattern_len = len(patron)
        
        # Asegura consistencia dimensional
        if len(patron) != self.pattern_len:
            p_list = list(patron) + [0] * (self.pattern_len - len(patron))
            patron = tuple(p_list[:self.pattern_len])

        self.data.append(patron)

        if len(self.data) > 200: # Limita la memoria para evitar sobrecarga
            self.data.pop(0)

        if len(self.data) % 50 == 0 and self.k < self.max_cat:
            self.k += 1
            # print(f"--- Agente intenta aumentar K a {self.k} ---")
        
        # --- CORRECCIÓN DE ROBUSTEZ 1: Comprobar antes de ajustar ---
        # Solo ajusta el modelo si tiene suficientes datos únicos
        unique_data = np.unique(np.array(self.data), axis=0)
        if len(unique_data) >= self.k:
            with warnings.catch_warnings():
                # Suprimimos las advertencias de convergencia que son esperables
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                self.model = KMeans(n_clusters=self.k, n_init='auto').fit(unique_data)

    def predecir_categoria(self, patron):
        if self.model is None or not patron:
            return 0
        
        if len(patron) != self.pattern_len:
            p_list = list(patron) + [0] * (self.pattern_len - len(patron))
            patron = tuple(p_list[:self.pattern_len])

        return int(self.model.predict([patron])[0])

# ------------------- CLASE 2: Neurona (con política reactiva simple) -------------------
class Neurona:
    def __init__(self, idx):
        self.id = idx
        self.estado = random.choice([0, 1])
        self.vecinos = []
        self.semantica = Semantica()
        self.Qval = {}
        self.kappa_pred = 1.0
        self.prox_estado = self.estado

    def set_vecinos(self, vecinos):
        self.vecinos = vecinos

    def percibir(self, neuronas):
        if not self.vecinos: return tuple()
        return tuple(sorted([neuronas[v].estado for v in self.vecinos]))

    def decidir(self, patron):
        cat = self.semantica.predecir_categoria(patron)
        q_val = self.Qval.get(cat, 0.0)
        prob_activacion = 0.5 + q_val * 0.5
        self.prox_estado = 1 if random.random() < prob_activacion else 0
        
    def aprender(self, patron_previo, neuronas):
        if not self.vecinos or not patron_previo:
            self.kappa_pred = 0
            return
            
        self.semantica.actualizar(patron_previo)
        
        estados_reales_vecinos = tuple(sorted([neuronas[v].estado for v in self.vecinos]))
        cat_real = self.semantica.predecir_categoria(estados_reales_vecinos)
        q_val_real = self.Qval.get(cat_real, 0.0)
        prediccion_ideal = 1 if q_val_real > 0 else 0
        
        estado_real_agregado = 1 if sum(n.estado for n in [neuronas[v] for v in self.vecinos]) > len(self.vecinos) / 2 else 0
        
        error_pred = abs(prediccion_ideal - estado_real_agregado)
        self.kappa_pred = 0.9 * self.kappa_pred + 0.1 * error_pred

        r_val = -self.kappa_pred
        cat_previa = self.semantica.predecir_categoria(patron_previo)
        self.Qval.setdefault(cat_previa, 0.0)
        self.Qval[cat_previa] += LR_VAL * (r_val - self.Qval[cat_previa])
        
    def actualizar(self):
        self.estado = self.prox_estado

# ------------------- CLASE 3: Mundo (Con Grafo ESTÁTICO) -------------------
class Mundo:
    def __init__(self, n, p):
        self.n = n
        self.grafo = nx.gnp_random_graph(n, p)
        self.neuronas = {i: Neurona(i) for i in range(n)}
        for i in range(n):
            self.neuronas[i].set_vecinos(list(self.grafo.neighbors(i)))

    def paso(self):
        patrones_previos = {}
        for i, n in self.neuronas.items():
            patron = n.percibir(self.neuronas)
            patrones_previos[i] = patron
            n.decidir(patron)
        
        for n in self.neuronas.values(): n.actualizar()
        for i, n in self.neuronas.items(): n.aprender(patrones_previos[i], self.neuronas)
            
    def medida_global(self):
        kappas_p = [n.kappa_pred for n in self.neuronas.values()]
        k_values = [n.semantica.k for n in self.neuronas.values()]
        return np.mean(kappas_p), np.mean(k_values), np.max(k_values)

# ------------------- SIMULACIÓN PRINCIPAL -------------------
if __name__ == '__main__':
    m = Mundo(N_NEURONAS, CONECTIVIDAD)
    hist = {'kp':[], 'k_mean':[], 'k_max':[]}

    print(">> INICIANDO SIMULACIÓN (Etapa 1.1 - Semántica Robusta) <<\n")
    for t in range(TMAX):
        m.paso()
        if t % 50 == 0 or t == TMAX - 1:
            kp, k_mean, k_max = m.medida_global()
            hist['kp'].append(kp)
            hist['k_mean'].append(k_mean)
            hist['k_max'].append(k_max)
            print(f"t={t:4d} | κ_pred_global={kp:.4f} | K Promedio={k_mean:.2f} | K Máximo={k_max}")

    # --- GRAFICAR RESULTADOS ---
    fig, ax1 = plt.subplots(figsize=(14, 7))

    color = 'tab:blue'
    ax1.set_xlabel('Paso de tiempo')
    ax1.set_ylabel('κ_pred Global (Error Predictivo)', color=color)
    ax1.plot([i*50 for i in range(len(hist['kp']))], hist['kp'], color=color, marker='o', label='κ_pred (Error L1)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Complejidad Semántica (Nivel de K)', color=color)
    ax2.plot([i*50 for i in range(len(hist['k_mean']))], hist['k_mean'], color=color, linestyle='--', label='K Promedio')
    ax2.plot([i*50 for i in range(len(hist['k_max']))], hist['k_max'], color='tab:green', linestyle=':', label='K Máximo')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')
    
    fig.tight_layout()
    plt.title('Evolución del Error Predictivo vs. la Complejidad Semántica Emergente')
    plt.show()