import numpy as np
from matplotlib import pyplot as plt


class AnomalyProcessing:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.ylim = (np.min(y), np.max(y))

        self.size_of_window = None

    def _process_anomaly(self, anomaly_indices, method='remove'):
        print(f'Знайдено {len(anomaly_indices)} аномалій.')
        if len(anomaly_indices) == 0:
            return

        if method == 'remove':
            self.x = np.delete(self.x, anomaly_indices)
            self.y = np.delete(self.y, anomaly_indices)
        elif method == 'mean':
            # mean 2 neighbors, but if they are not in anomalies
            for i in anomaly_indices:
                if i > 0 and i < len(self.y) - 1:
                    self.y[i] = (self.y[i - 1] + self.y[i + 1]) / 2
                elif i == 0:
                    self.y[i] = self.y[i + 1]
                else:
                    self.y[i] = self.y[i - 1]


    def _find_best_params(self):
        y = self.y
        max_window = min(50, len(y) // 4)  # обмеження на великі вікна
        candidate_windows = list(range(5, max_window, 2))  # тільки непарні розміри вікна
        candidate_coeffs = np.arange(1.0, max(3.0, max(self.y) / min(self.y)), 0.25)  # коефіцієнти від 1 до 3 з кроком 0.25

        best_score = float('inf')
        best_params = (5, 2.0)

        for w in candidate_windows:
            rolling_mean = np.convolve(y, np.ones(w) / w, mode='valid')
            rolling_std = np.array([
                np.std(y[i:i + w]) for i in range(len(y) - w + 1)
            ])
            for coeff in candidate_coeffs:
                z_scores = np.abs((y[w - 1:] - rolling_mean) / rolling_std)
                anomalies = np.where(z_scores > coeff)[0]

                # Оцінка якості: компроміс між кількістю аномалій та середнім std
                num_anomalies = len(anomalies)
                mean_std = np.mean(rolling_std)
                score = mean_std * (1 + num_anomalies / len(y))  # чим менше — тим краще

                if score < best_score and 0 < num_anomalies < len(y) * 0.3:
                    best_score = score
                    best_params = (w, coeff)

        print(f'Обрано параметри: вікно={best_params[0]}, коефіцієнт={best_params[1]}')
        return best_params


    def _find_anomalies_medium(self):
        y = self.y.copy()
        Nwin, Q = self._find_best_params()

        anomalies = []

        # Еталонне вікно
        x_ref = y[:Nwin]
        std_ref = np.std(x_ref)

        for i in range(Nwin, len(y)):
            # Поточне вікно
            current_window = y[i - Nwin + 1: i + 1]
            std_current = np.std(current_window)

            if std_ref > 0 and std_current > Q * std_ref:
                anomalies.append(i)

        return anomalies


    def _draw_new_data(self, method='remove'):
        plt.figure(figsize=(20, 10))
        plt.plot(self.x, self.y, label='Оброблені дані')
        plt.title(f'Оновлені дані після обробки аномалій (за методом medium, метод обробки - {method})')
        plt.ylim(self.ylim)
        plt.legend()
        plt.show()


    def start_processing(self, method='remove'):
        anomalies = self._find_anomalies_medium()
        self._process_anomaly(anomalies, method)
        self._draw_new_data(method)

