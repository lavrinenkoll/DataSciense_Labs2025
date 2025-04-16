'''
1. Провести парсинг самостійно обраного сайту. Вміст даних, що підлягають парсингу – обрати самостійно.
2. Результати парсингу зберегти у файлі. Тип файлу обрати самостійно.

3. Оцінити динаміку тренду реальних даних.
4. Здійснити визначення статистичних характеристик результатів парсингу.

5. Синтезувати та верифікувати модель даних, аналогічних за трендом і статистичними характеристиками реальним даним, які є результатом парсингу.
6. Провести аналіз отриманих результатів.
'''
import pandas as pd
from lab1.parser import Parser
from lab1.statistical import NUMBER_OF_SEPARATION_LINES
from lab1.tools import results_trend
from lab1.synthetic_data import SyntheticData

MAX_DEGREE_POLYNOMIAL = 100

def process_data(data_x, data_y, title_x, title_y, max_degree=30, noise_factor=0.05, number_x=200):
    datastats = results_trend(data_x, data_y, title_x, title_y, max_degree)

    synthetic_data = SyntheticData(datastats.x, datastats.y, datastats.intercept, datastats.coef, datastats.model, title_y,
                                      noise_factor=noise_factor, number_x=number_x, type_of_noise='uniform')
    synthetic_data.start_generation()

    synthetic_data = SyntheticData(datastats.x, datastats.y, datastats.intercept, datastats.coef, datastats.model, title_y,
                                      noise_factor=noise_factor, number_x=number_x, type_of_noise='normal')
    synthetic_data.start_generation()

    return datastats, synthetic_data


if __name__ == '__main__':
    # parse data from ekatalog website, save to files
    # https://ek.ua/ua/ECOFLOW-DELTA-2.htm
    url = input('Введіть URL для парсингу (сторінка товару на ekatalog): ')
    if not url:
        url = 'https://ek.ua/ua/CANON-50MM-F-1-4-EF-USM.htm'
    parser = Parser(url)
    price_change, number_of_shops = parser.parse_table(save_to_files=True)

    # read from file, if needed
    # price_change = pd.read_excel('./data/ECOFLOW-DELTA-2_price_change.xlsx')
    # number_of_shops = pd.read_excel('./data/ECOFLOW-DELTA-2_number_of_shops.xlsx')

    noise = input('Введіть фактор шуму (від 0 до 1, за замовчуванням 0.05): ')
    number_x = input(f'Введіть кількість значень для синтетичних даних (за замовчуванням як в оригінальних: {len(price_change)}): ')
    degree = input(f'Введіть максимальний степінь пошуку найкращого полінома (ціле число більше 1, за замовчуванням: {MAX_DEGREE_POLYNOMIAL}): ')
    print('-' * NUMBER_OF_SEPARATION_LINES)
    noise_factor = float(noise) if noise else 0.05
    number_x = int(number_x) if number_x else len(price_change)
    degree = int(degree) if degree else MAX_DEGREE_POLYNOMIAL
    if degree < 1:
        degree = 1


    process_data(price_change['date'], price_change['avg'], 'Дата', 'Середня ціна',
                 max_degree=degree, noise_factor=noise_factor, number_x=number_x)
    process_data(price_change['date'], price_change['min'], 'Дата', 'Мінімальна ціна',
                 max_degree=degree, noise_factor=noise_factor, number_x=number_x)
    process_data(price_change['date'], price_change['max'], 'Дата', 'Максимальна ціна',
                 max_degree=degree, noise_factor=noise_factor, number_x=number_x)
    process_data(number_of_shops['date'], number_of_shops['value'], 'Дата', 'Кількість магазинів',
                 max_degree=degree, noise_factor=noise_factor, number_x=number_x)


