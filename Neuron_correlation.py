# Numpy realization

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
import nest
nest.set_verbosity("M_ERROR")

def make_phi_theta(p, c):
    phi = p * (1 - c ** (1 / 2))
    theta = p + (1 - p) * c ** (1 / 2)

    return phi, theta

# Создается эталонная последовательность
def standart_list(p, N):
    S0 = np.random.rand(N)

    return np.where(S0 < p, 1, 0)

# От эталонной последовательности создаются наборы коррелирующих производных
# последовательностей для каждого элемента C_vect
def make_correlation_S0_Ymatrix(p, C_vect, N, K):
    S0 = standart_list(p, N)
    # Если эталонная последовательность не содердит единиц-'спайков', то
    # предлагается изменить входные параметры и перезапустить функцию
    if not (1 in S0):
        print('Not any spikes in standart list - Try to change p or N in corr_params')
        return -1, -1

    # Содержит список наборов коррелирующих последовательностей
    Ymatrix = np.random.rand(len(C_vect), K, N)

    for i in range(len(C_vect)):
        c = C_vect[i]
        phi, theta = make_phi_theta(p, c)

        for j in range(K):
            Ymatrix[i, j] = np.where((Ymatrix[i, j] < S0) & (Ymatrix[i, j] < theta) |
                                     (Ymatrix[i, j] >= S0) & (Ymatrix[i, j] < phi),
                                     1, 0
                                     )

    return S0, Ymatrix


# Настройка нейрона
def create_neuron(neuron_params):
    neuron = nest.Create(neuron_params['neuron_model'])
    neuron.set(neuron_params['neuron_set'])

    return neuron

# Генератор спайков создает новые спайки от новых кор. произв. послед. с учетом времени симмуляции
def create_spike_times(time_list, Ymatrix, time):
    d = 0
    for i in range(len(Ymatrix)):
        for j in range(len(Ymatrix[0])):
            time_list[d]['spike_times'] = (((np.where(Ymatrix[i, j] == 1)[0]) * 0.1 + 0.1).round(1) + time).tolist()
            d += 1

    return time_list


def make_time_list(m):
    time_list = []
    for i in range(m):
        time_list.append({'spike_times': []})

    return time_list

# Создание нейрона и топологии сети, симуляция и получение весов stdp соединений
def learning_network(X, p, N, K, neuron_params, synapse_params):
    nest.ResetKernel()

    neuron = create_neuron(neuron_params)

    # len(X[0]) - длина 1го вектора Х
    m = K * len(X[0])

    all_spike_generators = nest.Create("spike_generator", m)

    parrot_pop = nest.Create("parrot_neuron", m)
    nest.Connect(all_spike_generators, parrot_pop, 'one_to_one')
    nest.Connect(parrot_pop, neuron, syn_spec=synapse_params)

    Connections = nest.GetConnections(parrot_pop, neuron)

    time_list = make_time_list(m)

    time_simulation = 0.1 * N  # время симмуляции

    time = 0  # Общее время (всех симмуляций)

    for X_i in X:
        C_vect = X_i
        # S0 - эталонная последовательность
        # Ymatrix - список, содержащий списки производных последовательностей
        S0, Ymatrix = make_correlation_S0_Ymatrix(p, C_vect, N, K)

        time_list = create_spike_times(time_list, Ymatrix, time)

        nest.SetStatus(all_spike_generators, time_list)

        time += time_simulation

        nest.Simulate(time_simulation)

    Weights = Connections.get('weight')

    nest.ResetKernel()

    return Weights


# Массив спайков на выходе нейрона после симуляции
def get_out_spikes(dSD, l):
    out_spikes = np.zeros(l)

    ts = dSD["times"]
    out_spikes[((ts - 0.1) / 0.1).astype('int32')] = 1

    return out_spikes


# Кросс-корреляция
def cross_correlation(S0, out_spikes, N, cross_cor_n):
    Out_i = 0
    for i in range(cross_cor_n + 1):
        Out_i += np.sum(out_spikes[i: N + i] * S0)

    return Out_i


# Выводит кросс-корреляцию, cоздает нейрон и топологию сети
def predicting_network(X, Weights, p, N, K, cross_cor_n, neuron_params):
    nest.ResetKernel()

    m = K * len(X[0])
    l = N + cross_cor_n

    Out = []

    for X_i in X:
        neuron = create_neuron(neuron_params)

        spikerecorder = nest.Create("spike_recorder")
        nest.Connect(neuron, spikerecorder)

        C_vect = X_i
        # S0 - эталонная последовательность
        # Ymatrix - список, содержащий наборы производных последовательностей
        S0, Ymatrix = make_correlation_S0_Ymatrix(p, C_vect, N, K)

        time_list = make_time_list(m)
        time_list = create_spike_times(time_list, Ymatrix, 0)

        all_spike_generators = nest.Create("spike_generator", m)

        syn_dict = {"synapse_model": "static_synapse", "weight": [Weights]}
        nest.Connect(all_spike_generators, neuron, syn_spec=syn_dict)

        nest.SetStatus(all_spike_generators, time_list)

        # Время симуляции
        time_simulation = 0.1 * l
        nest.Simulate(time_simulation)

        dSD = spikerecorder.get("events")
        out_spikes = get_out_spikes(dSD, l)

        # Кросс-корреляция S0 и out_spikes
        Out.append(cross_correlation(S0, out_spikes, N, cross_cor_n))

        nest.ResetKernel()

    return np.array(Out)


class CorrelationNetwork(BaseEstimator, ClassifierMixin):

    def __init__(self, corr_params, neuron_params, synapse_params):

        self.corr_params = corr_params
        self.neuron_params = neuron_params
        self.synapse_params = synapse_params

    def fit(self, X, y):

        if len(X) != len(y):
            print('ERROR: X and y must have the same size')
            return 0

        if isinstance(X, np.ndarray) == False:
            X = np.array(X)
        if isinstance(y, np.ndarray) == False:
            y = np.array(y)

        self.cross_cor_n_ = np.round(self.corr_params['cross_cor_time'] / 0.1).astype('int32')

        # Содержит список наборов весов от разных классов
        self.weights_ = []

        self.train_out_ = []

        for current_class in set(y):
            current_class_X = X[y == current_class]

            current_class_weights = learning_network(current_class_X,
                                                     self.corr_params['p'],
                                                     self.corr_params['N'],
                                                     self.corr_params['K'],
                                                     self.neuron_params,
                                                     self.synapse_params)
            self.weights_.append(current_class_weights)

            current_class_out = predicting_network(current_class_X,
                                                   current_class_weights,
                                                   self.corr_params['p'],
                                                   self.corr_params['N'],
                                                   self.corr_params['K'],
                                                   self.cross_cor_n_,
                                                   self.neuron_params)
            self.train_out_.append(current_class_out)

        return self

    def predict_proba(self, X):

        all_classes_P = []
        for current_class_weights, current_class_out in zip(self.weights_, self.train_out_):
            current_class_weights_out = predicting_network(X,
                                                           current_class_weights,
                                                           self.corr_params['p'],
                                                           self.corr_params['N'],
                                                           self.corr_params['K'],
                                                           self.cross_cor_n_,
                                                           self.neuron_params)

            mean = np.mean(current_class_out)
            std = np.std(current_class_out)

            current_class_p = np.e ** (-((current_class_weights_out - mean) / std) ** 2)

            all_classes_P.append(current_class_p)

        all_classes_P = np.array(all_classes_P)
        trans_all_classes_P = all_classes_P.transpose()

        self.prob_ = []
        self.pred_ = []
        for vect_p in trans_all_classes_P:
            max_p = np.max(vect_p)
            self.prob_.append(max_p)
            self.pred_.append(np.where(vect_p == max_p)[0][0])

        self.prob_ = np.array(self.prob_)
        self.pred_ = np.array(self.pred_)

        return self.pred_, self.prob_

    def transform(self, X=None):
        return X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        self.predict_proba(X)
        return self.pred_


# Класс предобработки входного вектора - раскладывает компоненты входного вектора по гауссовым полям
class GRF(BaseEstimator, TransformerMixin):
    def __init__(self, n=5, a=1, c=0.15):
        self.n = n
        self.a = a
        self.c = c

    def fit(self, X=None, y=None):

        self.b_ = 1 / (2 * (self.n))

        return self

    def transform(self, X=None):

        X_trans = []
        for x in X:
            x_trans = []
            for x_i in x:
                b = self.b_
                for j in range(self.n):
                    x_trans.append(self.a * np.e ** (-((x_i - b) ** 2) / (2 * (self.c) ** 2)))
                    b += (2 * self.b_)
            X_trans.append(x_trans)

        return np.array(X_trans)

    def fit_transform(self, X=None, y=None):
        self.fit(X, y)
        return self.transform(X)
