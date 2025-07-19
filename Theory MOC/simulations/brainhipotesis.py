import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import warnings
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from collections import defaultdict

# ------------------- PARÁMETROS GLOBALES DEL UNIVERSO -------------------
N_AGENTES    = 100
CONECTIVIDAD = 0.1
TMAX         = 3000
TAMANO_PERCEPCION = 15 

ETA_PLASTICIDAD_BASE = 0.01
LR_SEMANTICO         = 0.1 # Nombre unificado
UMBRAL_CRISIS_META   = 0.7
K_MAX_CATEGORIAS     = 8

# Parámetros del Meta-Agente (L3)
LR_META = 0.1
GAMMA_META = 0.9
EPSILON_META = 0.2

# Parámetros del Experimento de Trauma
TIEMPO_TRAUMA = 1500
PORCENTAJE_TRAUMA = 0.3

# ------------------- CAPA L1: FÍSICA Y ESTRUCTURA -------------------
class CapaFisica:
    def __init__(self, n, p):
        self.n = n
        self.estados = np.random.choice([0, 1], n)
        self.W = np.random.rand(n, n) * (np.random.rand(n, n) < p).astype(float)
        np.fill_diagonal(self.W, 0)
        self.grafo = nx.from_numpy_array(self.W)

    def obtener_vecinos(self, idx):
        return list(self.grafo.neighbors(idx))

    def aplicar_plasticidad(self, eta):
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.W[i, j] > 0:
                    exito = 1 if self.estados[i] == self.estados[j] else -1
                    cambio = eta * (exito * 0.1 - self.W[i, j] * 0.005) 
                    self.W[i, j] = max(0, min(1, self.W[i, j] + cambio))
                    self.W[j, i] = self.W[i, j]

    def reconstruir_grafo_desde_W(self):
        self.grafo = nx.from_numpy_array(self.W)

# ------------------- CAPA L2: SEMÁNTICA Y CONCIENCIA -------------------
class CapaSemantica:
    def __init__(self, n, k_max):
        self.n = n
        self.k_max = k_max
        self.modelos_k = {i: None for i in range(n)}
        self.datos_patrones = {i: [] for i in range(n)}
        self.k_actual = {i: 2 for i in range(n)}
        self.Q_semantico = {i: defaultdict(float) for i in range(n)}
        self.kappa_pred = np.ones(n)
        self.aprendizaje_congelado = False

    def _estandarizar_patron(self, patron):
        p_list = list(patron)
        if len(p_list) > TAMANO_PERCEPCION:
            return tuple(p_list[:TAMANO_PERCEPCION])
        else:
            return tuple(p_list + [0] * (TAMANO_PERCEPCION - len(p_list)))

    def actualizar_categorias(self, id_agente, patron):
        patron_estandar = self._estandarizar_patron(patron)
        self.datos_patrones[id_agente].append(patron_estandar)
        
        if len(self.datos_patrones[id_agente]) > 200: self.datos_patrones[id_agente].pop(0)

        if len(self.datos_patrones[id_agente]) % 100 == 0 and self.k_actual[id_agente] < self.k_max:
            self.k_actual[id_agente] += 1
        
        unique_data = np.unique(np.array(self.datos_patrones[id_agente]), axis=0)
        if len(unique_data) >= self.k_actual[id_agente]:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                self.modelos_k[id_agente] = KMeans(n_clusters=self.k_actual[id_agente], n_init='auto').fit(unique_data)

    def predecir_categoria(self, id_agente, patron):
        modelo = self.modelos_k[id_agente]
        if modelo is None or not hasattr(modelo, 'cluster_centers_') or not any(patron): return 0
        patron_estandar = self._estandarizar_patron(patron)
        return int(modelo.predict([patron_estandar])[0])

    def aprender_valor(self, id_agente, patron_previo, estados_vecinos_actual, lr_semantico):
        if self.aprendizaje_congelado or not patron_previo: return
        self.actualizar_categorias(id_agente, patron_previo)
        
        cat_previa = self.predecir_categoria(id_agente, patron_previo)
        cat_real = self.predecir_categoria(id_agente, tuple(sorted(estados_vecinos_actual)))
        q_val_real = self.Q_semantico[id_agente].get(cat_real, 0.0)
        prediccion_ideal = 1 if q_val_real > 0 else 0
        
        estado_real_agregado = 1 if sum(estados_vecinos_actual) > len(estados_vecinos_actual) / 2 else 0
        
        error = abs(prediccion_ideal - estado_real_agregado)
        self.kappa_pred[id_agente] = 0.9 * self.kappa_pred[id_agente] + 0.1 * error
        
        recompensa = -self.kappa_pred[id_agente]
        self.Q_semantico[id_agente][cat_previa] += lr_semantico * (recompensa - self.Q_semantico[id_agente][cat_previa])

# ------------------- CAPA L3: META-COGNICIÓN -------------------
class CapaMeta:
    def __init__(self, lr_meta, epsilon_meta):
        self.Q_meta = defaultdict(float)
        self.acciones = ['eta_up', 'eta_down', 'lr_up', 'lr_down', 'no_op']
        self.lr_meta = lr_meta
        self.gamma_meta = GAMMA_META
        self.epsilon_meta = epsilon_meta
        self.kappa_meta = 1.0

    def discretizar_estado(self, kappa_global, sigma_W):
        return (int(kappa_global * 15), int(sigma_W * 15))

    def elegir_accion(self, estado):
        if random.random() < self.epsilon_meta:
            return random.choice(self.acciones)
        q_vals = [self.Q_meta.get((estado, a), 0.0) for a in self.acciones]
        return self.acciones[np.argmax(q_vals)]

    def aprender(self, s, a, r, s_next):
        q_actual = self.Q_meta.get((s, a), 0.0)
        q_max_futuro = max([self.Q_meta.get((s_next, ac), 0.0) for ac in self.acciones])
        nuevo_q = q_actual + self.lr_meta * (r + self.gamma_meta * q_max_futuro - q_actual)
        self.Q_meta[(s, a)] = nuevo_q
        
# ------------------- CEREBRO V9.1 (con Trauma) -------------------
class Cerebro:
    def __init__(self, n, p, eta_base, lr_semantico_base):
        self.L1 = CapaFisica(n, p)
        self.L2 = CapaSemantica(n, K_MAX_CATEGORIAS)
        self.L3 = CapaMeta(LR_META, EPSILON_META)
        self.eta_actual = eta_base
        self.lr_semantico_actual = lr_semantico_base
        self.estado_meta_previo = None
        self.accion_meta_previa = None
        self.kappa_global_previo = 0.5
        self.presupuesto_energia = {"L1": 0.3, "L2": 0.6, "L3": 0.1}

    def _percibir_estandarizado(self, id_agente):
        vecinos_ids = self.L1.obtener_vecinos(id_agente)
        if not vecinos_ids: return tuple([0] * TAMANO_PERCEPCION)
        estados_vecinos = list(sorted(self.L1.estados[vecinos_ids]))
        if len(estados_vecinos) > TAMANO_PERCEPCION:
            return tuple(estados_vecinos[:TAMANO_PERCEPCION])
        else:
            return tuple(estados_vecinos + [0] * (TAMANO_PERCEPCION - len(estados_vecinos)))

    def paso_simulacion(self):
        patrones_previos = {}
        proximos_estados = np.zeros(self.L1.n, dtype=int)
        
        for i in range(self.L1.n):
            patron = self._percibir_estandarizado(i)
            patrones_previos[i] = patron
            cat = self.L2.predecir_categoria(i, patron)
            q_val = self.L2.Q_semantico[i].get(cat, 0.0)
            prob_activacion = 0.5 + q_val * 0.5
            proximos_estados[i] = 1 if random.random() < prob_activacion else 0
        
        self.L1.estados = proximos_estados

        if not self.L2.aprendizaje_congelado:
            for i in range(self.L1.n):
                vecinos_ids = self.L1.obtener_vecinos(i)
                estados_vecinos_actuales = list(self.L1.estados[vecinos_ids])
                self.L2.aprender_valor(i, patrones_previos[i], estados_vecinos_actuales, self.lr_semantico_actual * self.presupuesto_energia["L2"])
        
        self.L1.aplicar_plasticidad(self.eta_actual * self.presupuesto_energia["L1"])
        
        kappa_global_actual = np.mean(self.L2.kappa_pred)
        sigma_W_actual = np.std(self.L1.W[self.L1.W > 0]) if np.any(self.L1.W > 0) else 0
        self.L3.kappa_meta = 0.95 * self.L3.kappa_meta + 0.05 * abs(kappa_global_actual - self.kappa_global_previo)
        estado_meta_actual = self.L3.discretizar_estado(kappa_global_actual, sigma_W_actual)

        if self.estado_meta_previo is not None:
            recompensa_meta = -self.L3.kappa_meta * self.presupuesto_energia["L3"]
            self.L3.aprender(self.estado_meta_previo, self.accion_meta_previa, recompensa_meta, estado_meta_actual)
            
        accion_meta = self.L3.elegir_accion(estado_meta_actual)
        if accion_meta == 'eta_up': self.eta_actual *= 1.1
        if accion_meta == 'eta_down': self.eta_actual *= 0.9
        if accion_meta == 'lr_up': self.lr_semantico_actual *= 1.1
        if accion_meta == 'lr_down': self.lr_semantico_actual *= 0.9
        self.eta_actual = max(0.001, min(0.1, self.eta_actual))
        self.lr_semantico_actual = max(0.01, min(0.5, self.lr_semantico_actual))
        self.estado_meta_previo = estado_meta_actual
        self.accion_meta_previa = accion_meta
        self.kappa_global_previo = kappa_global_actual

        if self.L3.kappa_meta > UMBRAL_CRISIS_META:
            self.L2.aprendizaje_congelado = True
            self.presupuesto_energia = {"L1": 0.2, "L2": 0.1, "L3": 0.7}
        else:
            self.L2.aprendizaje_congelado = False
            self.presupuesto_energia = {"L1": 0.3, "L2": 0.6, "L3": 0.1}

    def aplicar_trauma(self, porcentaje):
        print("\n" + "*"*20 + " ¡TRAUMA ONTOLÓGICO! " + "*"*20)
        filas, columnas = np.where(self.L1.W > 0)
        conexiones = list(zip(filas, columnas))
        n_a_borrar = int(len(conexiones) * porcentaje)
        if n_a_borrar > 0 and len(conexiones) > n_a_borrar:
            conexiones_a_borrar = random.sample(conexiones, n_a_borrar)
            for u, v in conexiones_a_borrar:
                self.L1.W[u, v] = 0
                self.L1.W[v, u] = 0
        print(f"Se han destruido {len(conexiones_a_borrar)} conexiones.\n")
        self.L1.reconstruir_grafo_desde_W()

# ------------------- SIMULACIÓN PRINCIPAL -------------------
if __name__ == '__main__':
    cerebro = Cerebro(N_AGENTES, CONECTIVIDAD, ETA_PLASTICIDAD_BASE, LR_SEMANTICO)
    hist = {'t':[], 'k_pred':[], 'k_meta':[], 'q_val_meta':[]}
    
    print(">> INICIANDO SIMULACIÓN (V9.1 - El Desafío del Trauma) <<\n")
    for t in range(TMAX):
        if t == TIEMPO_TRAUMA:
            cerebro.aplicar_trauma(PORCENTAJE_TRAUMA)

        cerebro.paso_simulacion()
        
        if t % 100 == 0 or t == TMAX - 1:
            kp = np.mean(cerebro.L2.kappa_pred)
            km = cerebro.L3.kappa_meta
            q_val_meta_actual = cerebro.L3.Q_meta.get((cerebro.estado_meta_previo, cerebro.accion_meta_previa), 0.0)

            hist['t'].append(t)
            hist['k_pred'].append(kp)
            hist['k_meta'].append(km)
            hist['q_val_meta'].append(q_val_meta_actual)
            
            print(f"t={t:4d} | κ_pred={kp:.4f} | κ_meta={km:.4f} | Presupuesto L3: {cerebro.presupuesto_energia['L3']*100:.0f}%")

    # --- GRAFICAR RESULTADOS ---
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    timesteps_reported = hist['t']

    ax1.set_xlabel('Paso de tiempo')
    ax1.set_ylabel('Kappa (Error Predictivo)', color='blue')
    ax1.plot(timesteps_reported, hist['k_pred'], color='blue', marker='.', label='κ_pred (Error L2)')
    ax1.plot(timesteps_reported, hist['k_meta'], color='cyan', linestyle=':', label='κ_meta (Error L3)')
    ax1.axvline(x=TIEMPO_TRAUMA, color='black', linestyle='--', label='Evento de Trauma')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color = 'red'
    ax2.set_ylabel('Q-Valor del Meta-Agente (Política L3)', color=color)
    ax2.plot(timesteps_reported, hist['q_val_meta'], color=color, linestyle='--', label='Q-Valor L3')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')
    
    fig.tight_layout()
    plt.title('Dinámica de un Cerebro Especializado: Prueba de Antifragilidad')
    plt.show()