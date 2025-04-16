import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import math
from lab1.statistical import NUMBER_OF_SEPARATION_LINES
import textwrap

class SyntheticData:
    def __init__(self, data_x, data_y, intercept, coef, model, title, noise_factor=0.01, number_x=None, type_of_noise='uniform', random_seed=17):
        """
        Initialize the SyntheticData object with the original data and model.
        :param data_x: np.array
        :param data_y: np.array
        :param intercept: float
        :param coef: list
        :param model: sklearn model
        :param title: str
        :param noise_factor: float (default: 0.01)
        :param number_x: int (default: None)
        :param type_of_noise: str (default: 'uniform')
        :param random_seed: int (default: 17)
        """
        self.data_x = data_x
        self.data_y = data_y
        self.intercept = intercept
        self.coef = coef
        self.model = model
        self.title = title
        if type_of_noise not in ['uniform', 'normal']:
            raise ValueError('type_of_noise must be "uniform" or "normal"')
        self.type_of_noise = type_of_noise
        self.noise_factor = noise_factor
        if number_x is not None:
            self.number_x = number_x
        else:
            self.number_x = len(data_x)
        self.random_seed = random_seed

        self.x_synthetic = None
        self.y_synthetic = None
        self.y_synthetic_trend = None
        self.model_synthetic = None

    def statistics(self, print_stats=True, r2_print=True):
        """
        Calculate the statistical characteristics of the synthetic data.
        """
        stats = {
            'Середнє': np.mean(self.y_synthetic),
            'Медіана': np.median(self.y_synthetic),
            'Мінімум': np.min(self.y_synthetic),
            'Максимум': np.max(self.y_synthetic),
            'Стандартне відхилення': np.std(self.y_synthetic),
            'Дисперсія': np.var(self.y_synthetic),
            'Кількість записів': len(self.y_synthetic),
        }
        stats_trend = {
            'R²': self.model_synthetic.score(
                PolynomialFeatures(degree=len(self.coef)).fit_transform(self.x_synthetic.reshape(-1, 1)),
                self.y_synthetic),
            'MSE': mean_squared_error(self.y_synthetic, self.y_synthetic_trend),
            'RMSE': math.sqrt(mean_squared_error(self.y_synthetic, self.y_synthetic_trend))
        }

        if print_stats:
            print(f'\033[1m\033[4mСтатистичні характеристики для синтетичних даних (на основі {self.title.upper()}, '
                  f'закон розподілу шуму - {self.type_of_noise.upper()}):\033[0m')
            for key, value in stats.items():
                print(f'{key:<25}|{value:>20}')
            print('-' * NUMBER_OF_SEPARATION_LINES)
            print(f'\033[1m\033[4mОцінка тренду моделі синтетичних даних:\033[0m')
            for key, value in stats_trend.items():
                print(f'{key:<25}|{value:>20}')
            print()
            # Print the trend equation
            intercept = self.model_synthetic.intercept_
            coef = self.model_synthetic.coef_[1:]
            str_trend = f'Рівняння тренду: y = {intercept} + {" + ".join(f"{c} * x^{i + 1}" for i, c in enumerate(coef))}'
            print(textwrap.fill(str_trend, width=NUMBER_OF_SEPARATION_LINES-10))
            print()

        if r2_print:
            r_squared = stats_trend['R²']
            if r_squared > 0.9:
                print("R² показує, що модель тренду добре описує дані (понад 90% варіації даних пояснюється моделлю).")
            elif r_squared > 0.7:
                print(
                    "R² вказує на середній рівень відповідності моделі тренду даним (близько 70-90% варіації даних пояснюється моделлю).")
            else:
                print(
                    "R² показує, що модель тренду не дуже добре описує дані (менше 70% варіації даних пояснюється моделлю).")
            print('-' * NUMBER_OF_SEPARATION_LINES)

        return stats

    def _generate_data(self):
        """
        Generate synthetic data based on the original data.
        """
        np.random.seed(self.random_seed)
        self.x_synthetic = np.linspace(min(self.data_x), max(self.data_x), self.number_x)
        self.y_synthetic = self.intercept + sum(c * self.x_synthetic ** (i + 1) for i, c in enumerate(self.coef))

        if self.type_of_noise == 'uniform':
            self.y_synthetic += np.random.uniform(-max(self.data_y) * self.noise_factor,
                                                  max(self.data_y) * self.noise_factor, self.number_x)
        elif self.type_of_noise == 'normal':
            self.y_synthetic += np.random.normal(0, max(self.data_y) * self.noise_factor, self.number_x)

    def _fit_polynomial_model(self):
        """
        Learn the polynomial model for synthetic data.
        """
        degree = self.model.named_steps['polynomialfeatures'].degree if hasattr(self.model, 'named_steps') else 2
        poly = PolynomialFeatures(degree=degree)
        x_poly_synthetic = poly.fit_transform(self.x_synthetic.reshape(-1, 1))

        self.model_synthetic = LinearRegression()
        self.model_synthetic.fit(x_poly_synthetic, self.y_synthetic)
        self.y_synthetic_trend = self.model_synthetic.predict(x_poly_synthetic)

    def _plot_data(self):
        """
        Visualize the original and synthetic data trends.
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(self.x_synthetic, self.y_synthetic, color='blue', label='Синтетичні дані', s=10)
        # Trend line for original data (red)
        trend_line = self.intercept + sum(c * self.x_synthetic ** (i + 1) for i, c in enumerate(self.coef))
        plt.plot(self.x_synthetic, trend_line, color='red', label='Тренд оригінальних даних', lw=2)
        # Polynomial regression line for synthetic data
        plt.plot(self.x_synthetic, self.y_synthetic_trend, color='green', label='Тренд синтетичних даних', lw=2)
        plt.title(f'Порівняння трендів оригінальних ({self.title.upper()}) та синтетичних даних (тип шуму: {self.type_of_noise.upper()} з фактором {self.noise_factor})')
        plt.legend()
        plt.show()

    def start_generation(self, plot=True, stats=True, r2_print=True):
        self._generate_data()
        self._fit_polynomial_model()
        if plot:
            self._plot_data()
        self.statistics(stats, r2_print)
