import numpy as np
import matplotlib.pyplot as plt


class AnomalyGenerator:
    def __init__(self, number_of_anomalies, coef_anomalies, x, y, trend_y):
        self.number_of_anomalies = number_of_anomalies
        self.coef_anomalies = coef_anomalies
        self.x = x
        self.y = y
        self.trend_y = trend_y
        self.anomalies = self._generate_anomalies()

    def _generate_anomalies(self):
        self.anomalies = self.y.copy()
        number_of_anomalies = int(len(self.y) * self.number_of_anomalies)

        # randomly select indexes of anomalies
        indexes = np.random.choice(len(self.y), number_of_anomalies, replace=False)

        # generate anomalies, in right direction
        for i in indexes:
            if self.y[i] > self.trend_y[i]:
                self.anomalies[i] = self.y[i] * self.coef_anomalies
            else:
                self.anomalies[i] = self.y[i] / self.coef_anomalies

        return self.anomalies


    def draw_anomalies(self):
        plt.figure(figsize=(20, 10))
        plt.plot(self.x, self.trend_y, label='Тренд оригінальних даних')
        plt.plot(self.x, self.anomalies, label=f'Аномальні дані')
        plt.plot(self.x, self.y, label='Оригінальні дані')
        plt.title(f'Аномальні дані {self.number_of_anomalies*100}% від оригінальних, коефіцієнт {self.coef_anomalies}')
        plt.ylim(min(self.anomalies), max(self.anomalies))
        plt.legend()
        plt.show()
