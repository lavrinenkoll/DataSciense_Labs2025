import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import textwrap

NUMBER_OF_SEPARATION_LINES = 125


class DataStats:
    '''
    A class with x and y data for statistical analysis and plotting trends.
    '''

    def __init__(self, x, y, title_x='Date', title_y='Value', model=LinearRegression(), model_name='Лінійна регресія'):
        '''
        Initialize the DataStats object with x and y data.
        :param x: pd.Series
        :param y: pd.Series
        :param title_x: str
        :param title_y: str
        :param model: sklearn model (default: LinearRegression)
        '''
        self.x = x
        self.data_x = x # save original data for plotting
        self.y = y
        self.title_x = title_x
        self.title_y = title_y

        self.model = model
        self.model_name = model_name

        self.intercept = None
        self.coef = None

    def statistics(self, print_stats=False):
        '''
        Function to calculate the statistical characteristics of the data.
        :return: Dictionary with statistical characteristics
        '''
        stats = {
            'Середнє': np.mean(self.y),
            'Медіана': np.median(self.y),
            'Мінімум': np.min(self.y),
            'Максимум': np.max(self.y),
            'Стандартне відхилення': np.std(self.y),
            'Дисперсія': np.var(self.y),
            'Кількість записів': len(self.y),
        }

        if print_stats:
            print(f'\033[1m\033[4mСтатистичні характеристики для {self.title_y.upper()}:\033[0m')
            for key, value in stats.items():
                print(f'{key:<25}|{value:>20}')
            print('-' * NUMBER_OF_SEPARATION_LINES)
        return stats

    def _convert_dates_to_numeric(self):
        """
        Convert dates to numerical values (if x represents dates).
        :return: None
        """
        if pd.api.types.is_datetime64_any_dtype(self.x):
            date_min = self.x.min()
            self.x = (self.x - date_min) / np.timedelta64(1, 'D')  # Number of days from the minimum date

    def _fit_model(self):
        """
        Fit a regression model on x and y.
        :return: Trained regression model
        """
        X = self.x.values.reshape(-1, 1)  # Independent variable (number of days)
        y = self.y  # Dependent variable
        self.model.fit(X, y)
        return self.model

    def _predict_values(self):
        """
        Predict values using the trained model.
        :return: Predicted values
        """
        X = self.x.values.reshape(-1, 1)
        return self.model.predict(X)

    def _get_trend_equation(self):
        """
        Get the equation of the regression trend.
        :return: String representation of the equation
        """
        coef = self.model.named_steps['linearregression'].coef_ if hasattr(self.model,
                                                                           'named_steps') else self.model.coef_
        intercept = self.model.named_steps['linearregression'].intercept_ if hasattr(self.model,
                                                                                     'named_steps') else self.model.intercept_

        equation_terms = [f'{intercept}']
        for i, c in enumerate(coef[1:], start=1):
            equation_terms.append(f'{c} * x^{i}')

        equation = ' + '.join(equation_terms)

        self.intercept = intercept
        self.coef = coef[1:]

        return f'Рівняння тренду: y = {equation}'

    def _evaluate_trend(self, predicted, stats_r2_print=True):
        """
        Evaluate the trend by calculating R-squared, MSE, and RMSE.
        :param predicted: Predicted values
        :return: Dictionary with R², MSE, and RMSE
        """
        # Calculate R-squared
        r_squared = self.model.score(self.x.values.reshape(-1, 1), self.y)

        # Calculate MSE and RMSE
        mse = mean_squared_error(self.y, predicted)
        rmse = np.sqrt(mse)

        # Print evaluation metrics
        if stats_r2_print:
            print(f'\033[1m\033[4mОцінка тренду для {self.title_y.upper()}:\033[0m')
            print(f'R²{"":<23}|{r_squared:>20}')
            print(f'MSE{"":<22}|{mse:>20}')
            print(f'RMSE{"":<21}|{rmse:>20}')
            print()
            # get the trend equation
            print(textwrap.fill(self._get_trend_equation(), width=NUMBER_OF_SEPARATION_LINES-10))
            print()

            # Explanation of the metrics
            if r_squared > 0.9:
                print("R² показує, що модель тренду добре описує дані (понад 90% варіації даних пояснюється моделлю).")
            elif r_squared > 0.7:
                print(
                    "R² вказує на середній рівень відповідності моделі тренду даним (близько 70-90% варіації даних пояснюється моделлю).")
            else:
                print("R² показує, що модель тренду не дуже добре описує дані (менше 70% варіації даних пояснюється моделлю).")

            print('-' * NUMBER_OF_SEPARATION_LINES)

        return {'R2': r_squared, 'MSE': mse, 'RMSE': rmse}


    def _plot_graph(self, predicted):
        """
        Plot the trend graph with original and predicted data.
        :param predicted: Predicted values
        :return: None
        """
        plt.figure(figsize=(10, 6))

        plt.plot(self.data_x, self.y, label='Оригінальні дані', color='blue')
        plt.plot(self.data_x, predicted, label='Тренд', color='red')

        plt.xlabel(self.title_x)
        plt.ylabel(self.title_y)
        plt.title(f'Графік для {self.title_y.upper()} по {self.title_x.upper()}, використовуючи модель {self.model_name.upper()}')
        plt.legend()
        plt.show()

    def plot_trend(self, plot_need=True, stats_r2_print=True):
        """
        Function to plot a trend graph using regression by using modular helper functions.
        :return: None
        """
        self._convert_dates_to_numeric()
        self._fit_model()
        predicted = self._predict_values()
        if plot_need:
            self._plot_graph(predicted)
        dict_vals = self._evaluate_trend(predicted, stats_r2_print)

        return dict_vals


