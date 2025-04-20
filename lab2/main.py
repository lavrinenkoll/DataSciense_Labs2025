import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from lab1.parser import Parser
from lab1.statistical import NUMBER_OF_SEPARATION_LINES
from lab1.tools import results_trend
from lab1.synthetic_data import SyntheticData

from lab2.anomaly import AnomalyGenerator
from lab2.anomaly_processing import AnomalyProcessing

MAX_DEGREE_POLYNOMIAL = 100


def process_data(data_x, data_y, title_x, title_y, max_degree=30, noise_factor=0.05, number_x=200, type_of_noise='uniform'):
    datastats = results_trend(data_x, data_y, title_x, title_y, max_degree)

    if type_of_noise == 'uniform':
        synthetic_data = SyntheticData(datastats.x, datastats.y, datastats.intercept, datastats.coef, datastats.model,
                                       title_y,
                                       noise_factor=noise_factor, number_x=number_x, type_of_noise='uniform')
        synthetic_data.start_generation()

    else:
        synthetic_data = SyntheticData(datastats.x, datastats.y, datastats.intercept, datastats.coef, datastats.model,
                                       title_y,
                                       noise_factor=noise_factor, number_x=number_x, type_of_noise='normal')
        synthetic_data.start_generation()

    return datastats, synthetic_data


def quality_and_poly_model(x, original_data, cleaned_data, degree, plot=True):
    # Знаходимо спільну довжину
    min_len = min(len(original_data), len(cleaned_data), len(x))
    x = np.array(x[:min_len])
    y_orig = np.array(original_data[:min_len])
    y_clean = np.array(cleaned_data[:min_len])

    # Поліноміальні фічі
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(x.reshape(-1, 1))

    # Навчання моделей
    model_orig = LinearRegression().fit(X_poly, y_orig)
    model_clean = LinearRegression().fit(X_poly, y_clean)

    # Прогнози
    y_pred_orig = model_orig.predict(X_poly)
    y_pred_clean = model_clean.predict(X_poly)

    # Метрики
    mse = mean_squared_error(y_pred_orig, y_pred_clean)
    mae = mean_absolute_error(y_pred_orig, y_pred_clean)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_pred_orig, y_pred_clean)

    print('-' * NUMBER_OF_SEPARATION_LINES)
    print(f'🔎 Оцінка якості очищення (через моделі полінома ступеня {degree}):')
    print(f'MSE : {mse:.4f}')
    print(f'MAE : {mae:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'R²  : {r2:.4f}')
    print('-' * NUMBER_OF_SEPARATION_LINES)

    # Побудова графіка
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(x, y_clean, label='Очищені дані', alpha=0.5)
        plt.plot(x, y_orig, label='Оригінальні дані', alpha=0.5)
        plt.plot(x, y_pred_orig, label='Модель (оригінал)', linestyle='--')
        plt.plot(x, y_pred_clean, label='Модель (очищені)', linestyle='--')
        plt.legend()
        plt.title(f'Поліноміальне апроксимування (ступінь={degree})')
        plt.xlabel('Час')
        plt.ylabel('Значення')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return mse, mae, rmse, r2


def train_polynomial_model(x, y, degree=2):
    # Нормалізація для стабільного навчання
    x_mean, x_std = np.mean(x), np.std(x)
    x_norm = (x - x_mean) / x_std

    coeffs = np.polyfit(x_norm, y, degree)
    poly = np.poly1d(coeffs)

    # Формула
    formula = "y(t) = " + " + ".join([
        f"{c:.6f}*t^{deg}" for deg, c in zip(range(degree, -1, -1), coeffs)
    ])

    print("Коефіцієнти моделі:", coeffs)
    print("Рівняння (нормалізоване t):", formula)
    return poly, coeffs, formula, x_mean, x_std


def plot_polynomial_regression(x, y, poly, formula, x_mean, x_std):
    x_norm = (x - x_mean) / x_std
    y_pred = poly(x_norm)

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label="Вихідні дані")
    plt.plot(x, y_pred, 'r-', label="Поліноміальна регресія")
    plt.title("Поліноміальна регресія (МНК)\n")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_polynomial_extrapolation(x, y, poly, x_mean, x_std, interval=0.5):
    n = len(x)
    delta = x[1] - x[0]
    new_points = int(n * interval)
    x_ext = np.linspace(x[0], x[-1] + delta * new_points, n + new_points)
    x_ext_norm = (x_ext - x_mean) / x_std
    y_ext = poly(x_ext_norm)

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label="Вихідні дані")
    plt.plot(x_ext, y_ext, 'r--', label="Прогноз (екстраполяція)")
    plt.axvline(x[-1], color='gray', linestyle='--', label="Край спостереження")
    plt.title(f"Прогнозування (екстраполяція) на {interval} інтервалу")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()


def alpha_beta_filter(x, y):
    T0 = 1
    n = len(y)
    y = np.array(y).reshape(-1, 1)
    x = np.array(x).reshape(-1, 1)

    y_filtered = np.zeros_like(y)

    # Початкові умови
    speed_est = (y[1, 0] - y[0, 0]) / T0
    y_pred = y[0, 0] + speed_est

    y_filtered[0, 0] = y[0, 0]

    for i in range(1, n):
        # Динамічне оновлення коефіцієнтів
        alpha = (2 * (2 * i - 1)) / (i * (i + 1))
        beta = 6 / (i * (i + 1))

        # Помилка між виміром і прогнозом
        error = y[i, 0] - y_pred

        # Захист від розбіжності: обмежуємо помилку
        max_error = 3 * np.std(y[:i]) if i > 10 else 1e4
        if abs(error) > max_error:
            error = np.sign(error) * max_error

        # Оновлення згладженого значення
        y_filtered[i, 0] = y_pred + alpha * error

        # Оновлення швидкості
        speed_est = speed_est + (beta / T0) * error

        # Новий прогноз
        y_pred = y_filtered[i, 0] + speed_est

    # Побудова графіка
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Оригінальні дані', alpha=0.6)
    plt.plot(x, y_filtered, label='Після рекурсивного згладжування (α-β фільтр)', linewidth=2)
    plt.title("Рекурсивне згладжування за допомогою α-β фільтра")
    plt.xlabel("Час")
    plt.ylabel("Значення")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Статистичні показники
    print('Статистичні показники для згладжених даних:')
    print(f'Середнє: {np.mean(y_filtered):.2f}')
    print(f'Медіана: {np.median(y_filtered):.2f}')
    print(f'Мінімум: {np.min(y_filtered):.2f}')
    print(f'Максимум: {np.max(y_filtered):.2f}')
    print(f'Стандартне відхилення: {np.std(y_filtered):.2f}')
    print(f'Дисперсія: {np.var(y_filtered):.2f}')
    print(f'Кількість записів: {len(y_filtered)}')

    # r2 відповідність
    r_squared = r2_score(y, y_filtered)
    print(f'R²: {r_squared:.4f}')

    print('-' * NUMBER_OF_SEPARATION_LINES)

    return y_filtered


if __name__ == '__main__':
    # parse data from ekatalog website, save to files
    # https://ek.ua/ua/ECOFLOW-DELTA-2.htm
    url = input('Введіть URL для парсингу (сторінка товару на ekatalog): ')
    if not url:
        url = 'https://ek.ua/ua/CANON-50MM-F-1-4-EF-USM.htm'
    #url = 'https://ek.ua/ua/ECOFLOW-DELTA-2.htm'
    parser = Parser(url)
    price_change, number_of_shops = parser.parse_table(save_to_files=False)

    # read from file, if needed
    # price_change = pd.read_excel('./data/ECOFLOW-DELTA-2_price_change.xlsx')
    # number_of_shops = pd.read_excel('./data/ECOFLOW-DELTA-2_number_of_shops.xlsx')

    type_of_noise = input('Введіть тип шуму (uniform або normal, за замовчуванням normal): ')
    if type_of_noise not in ['uniform', 'normal']:
        type_of_noise = 'normal'
    noise = input('Введіть фактор шуму (від 0 до 1, за замовчуванням 0.05): ')
    number_x = input(
        f'Введіть кількість значень для синтетичних даних (за замовчуванням 10*оригінальні: {10*len(price_change)}): ')

    print('-' * NUMBER_OF_SEPARATION_LINES)
    noise_factor = float(noise) if noise else 0.05
    number_x = int(number_x) if number_x else 10*len(price_change)
    degree = MAX_DEGREE_POLYNOMIAL


    datastats, synthetic_data = process_data(price_change['date'], price_change['avg'], 'Дата', 'Середня ціна',
                 max_degree=degree, noise_factor=noise_factor, number_x=number_x, type_of_noise=type_of_noise)

    # datastats, synthetic_data = process_data(price_change['date'], price_change['max'], 'Дата', 'Максимальна ціна',
    #                                             max_degree=degree, noise_factor=noise_factor, number_x=number_x,
    #                                             type_of_noise=type_of_noise)

    # datastats, synthetic_data = process_data(price_change['date'], price_change['min'], 'Дата', 'Мінімальна ціна',
    #                                             max_degree=degree, noise_factor=noise_factor, number_x=number_x,
    #                                             type_of_noise=type_of_noise)


    # anomalyes
    number_of_anomalies = input('Введіть кількість аномальних вимірів у відсотках (за замовчуванням 5%): ')
    number_of_anomalies = int(number_of_anomalies)/100 if number_of_anomalies else 0.05
    coef_anomalies = input('Введіть коефіцієнт аномальних вимірів (за замовчуванням 1.25): ')
    coef_anomalies = float(coef_anomalies) if coef_anomalies else 1.25

    anomaly_generator = AnomalyGenerator(number_of_anomalies, coef_anomalies, synthetic_data.x_synthetic,
                                            synthetic_data.y_synthetic, synthetic_data.y_synthetic_trend)
    anomaly_generator.draw_anomalies()

    # anomaly processing
    print('-' * NUMBER_OF_SEPARATION_LINES)
    print('Обробка аномалій (метод усунення - видалення)')
    print('-' * NUMBER_OF_SEPARATION_LINES)
    anomaly_processing = AnomalyProcessing(anomaly_generator.x, anomaly_generator.anomalies)
    anomaly_processing.start_processing(method='remove')


    # Визначення показників якості видалених аномалій, порівняти з оригінальними даними
    quality_and_poly_model(synthetic_data.x_synthetic, synthetic_data.y_synthetic,
                               anomaly_processing.y, degree=len(synthetic_data.coef), plot=True)


    print('-' * NUMBER_OF_SEPARATION_LINES)
    print('Обробка аномалій (метод усунення - середнє)')
    print('-' * NUMBER_OF_SEPARATION_LINES)
    anomaly_processing = AnomalyProcessing(anomaly_generator.x, anomaly_generator.anomalies)
    anomaly_processing.start_processing(method='mean')

    # Визначення показників якості видалених аномалій, порівняти з оригінальними даними
    quality_and_poly_model(synthetic_data.x_synthetic, synthetic_data.y_synthetic,
                           anomaly_processing.y, degree=len(synthetic_data.coef), plot=True)

    # Реалізувати рекурентне згладжування alfa-beta, або alfa-beta-gamma фільтром сформованих в п.1, 2 вхідних даних. Прийняти заходи подолання
    # явища «розбіжності» фільтра. вхідні дані – anomaly_processing.x, anomaly_processing.y
    alpha_beta_filter(anomaly_processing.x, anomaly_processing.y)


    # Статистичне навчання поліноміальної моделі за методом найменших квадратів
    # (МНК – LSM) – поліноміальна регресія для вхідних даних, отриманих в п.1,2. - synthetic_data.x, synthetic_data.y
    # Прогнозування (екстраполяцію) параметрів досліджуваного процесу за «навченою»
    # моделлю на 0,5 інтервалу спостереження (об’єму вибірки);
    x, y, degree = synthetic_data.x_synthetic, synthetic_data.y_synthetic, len(synthetic_data.coef)
    poly, coeffs, formula, x_mean, x_std = train_polynomial_model(x, y, degree=degree)
    plot_polynomial_regression(x, y, poly, formula, x_mean, x_std)
    plot_polynomial_extrapolation(x, y, poly, x_mean, x_std, interval=0.05)
    plot_polynomial_extrapolation(x, y, poly, x_mean, x_std, interval=0.5)

