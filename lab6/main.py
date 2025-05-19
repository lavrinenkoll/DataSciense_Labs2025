from lab1.parser import Parser
from lab1.statistical import DataStats
from lab1.synthetic_data import SyntheticData
from lab1.tools import find_best_model_trend
from lab1.main import NUMBER_OF_SEPARATION_LINES

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def predict_with_lstm(data,
                      window_size=10,
                      percent_predict=0.1,
                      n_predictions = None,
                      epochs=20,
                      batch_size=16,
                      draw_graph=True,
                      print_data=True):

    scaler = StandardScaler()
    data = np.array(data).reshape(-1, 1)
    scaled = scaler.fit_transform(data)

    if n_predictions is None:
        n_predictions = int(percent_predict * len(data))

    X, y = [], []
    for i in range(len(scaled) - window_size):
        X.append(scaled[i:i + window_size])
        y.append(scaled[i + window_size])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=(window_size, 1)),
        LSTM(32, return_sequences=True, activation='relu'),
        LSTM(16, return_sequences=True, activation='relu'),
        LSTM(8, return_sequences=False, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    predictions = []
    last_window = scaled[-window_size:]
    for _ in range(n_predictions):
        pred = model.predict(last_window.reshape((1, window_size, 1)), verbose=0)
        predictions.append(pred[0][0])
        last_window = np.append(last_window[1:], pred[0][0])
        last_window = last_window.reshape((window_size, 1))
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    # Графік
    if draw_graph:
        plt.plot(np.arange(len(data)), data, label='Оригінальні дані', color='blue')
        plt.plot(np.arange(len(data), len(data) + n_predictions), predictions,
                 label='Прогнозовані дані', color='red')
        plt.legend()
        plt.title("Прогноз цін з використанням LSTM")
        plt.show()

    if print_data:
        print(f"Прогнозовані ціни на наступні {n_predictions} періодів:")
        print(' '.join(map(str, predictions.flatten())))
        print('-' * NUMBER_OF_SEPARATION_LINES)

    return predictions.flatten()


def cropped_predictions(data, percent_crop=0.1):
    n_predictions = int(percent_crop * len(data))
    data_cropped = data[:-n_predictions]
    cropped_end = data[-n_predictions:]

    predictions = predict_with_lstm(
        data_cropped,
        window_size=10,
        percent_predict=percent_crop,
        n_predictions=n_predictions,
        epochs=20,
        batch_size=16,
        draw_graph=False,
        print_data=False
    )

    plt.plot(np.arange(len(data_cropped)), data_cropped, label='Оригінальні дані', color='blue')
    plt.plot(np.arange(len(data_cropped), len(data_cropped) + n_predictions), predictions,
             label='Прогнозовані дані', color='red')
    plt.plot(np.arange(len(data_cropped), len(data_cropped) + n_predictions), cropped_end,
             label='Обрізані справжні', color='green')
    plt.legend()
    plt.title(f"Прогноз цін з використанням LSTM (обрізані дані {percent_crop * 100:.0f}%)")
    plt.grid(True)
    plt.show()

    print(f"Прогнозовані ціни на наступні {n_predictions} періодів:")
    print(' '.join(f"{v:.2f}" for v in predictions))
    print('Обрізані справжні значення:')
    print(' '.join(f"{v:.2f}" for v in cropped_end))
    print('-' * NUMBER_OF_SEPARATION_LINES)

    error = np.abs(predictions - cropped_end)
    print(f"Середня похибка: {np.mean(error):.2f}")
    print(f"Максимальна похибка: {np.max(error):.2f}")
    print(f"Мінімальна похибка: {np.min(error):.2f}")
    percent_error = (error / cropped_end) * 100
    print(f"Середня відсоткова похибка: {np.mean(percent_error):.2f}%")
    print('-' * NUMBER_OF_SEPARATION_LINES)

    return predictions


if __name__ == "__main__":
    url = input('Введіть URL для парсингу (сторінка товару на ekatalog): ')
    if not url:
        print('URL не введено. Використовується URL за замовчуванням.')
        url = 'https://ek.ua/ua/APPLE-IPHONE-16-PRO-MAX-256GB.htm'
    print('-' * NUMBER_OF_SEPARATION_LINES)

    parser = Parser(url)
    price_change, number_of_shops = parser.parse_table(save_to_files=False)

    x_analysis, x_title = price_change['date'], 'Дата'
    y_analysis, y_title = price_change['avg'], 'Середня ціна'

    best_model, max_r2, datastats_old = find_best_model_trend(x_analysis,
                                                                y_analysis,
                                                                title_x=x_title,
                                                                title_y=y_title,
                                                                max_degree=20)
    datastats = DataStats(x_analysis,
                            y_analysis,
                            title_x=x_title,
                            title_y=y_title,
                            model=best_model,
                            model_name=datastats_old.model_name)
    datastats.statistics(print_stats=True)
    datastats.plot_trend(stats_r2_print=True)

    synthetic_data = SyntheticData(datastats.x, datastats.y,
                                    datastats.intercept,
                                    datastats.coef,
                                    datastats.model,
                                    title=y_title,
                                    noise_factor=0.01,
                                    number_x=len(price_change)*2,
                                    type_of_noise='normal')

    synthetic_data.start_generation(plot=True, stats=True, r2_print=False)
    print('-' * NUMBER_OF_SEPARATION_LINES)
    percent_predict = input('Введіть відсоток прогнозування (за замовчуванням 10%): ')
    if not percent_predict:
        percent_predict = 0.1
    else:
        percent_predict = float(int(percent_predict)/100)

    print(f'Відсоток прогнозування: {percent_predict*100}%')
    print('-' * NUMBER_OF_SEPARATION_LINES)
    predictions = predict_with_lstm(
        data=synthetic_data.y_synthetic,
        window_size=10,
        percent_predict=percent_predict,
        epochs=20,
        batch_size=16
    )

    croped_percent = input('Введіть відсоток обрізки (за замовчуванням 10%): ')
    if not croped_percent:
        croped_percent = 0.1
    else:
        croped_percent = float(int(croped_percent)/100)
    print(f'Відсоток обрізки: {croped_percent*100}%')
    print('-' * NUMBER_OF_SEPARATION_LINES)
    cropped_predictions(
        data=synthetic_data.y_synthetic,
        percent_crop=croped_percent
    )
