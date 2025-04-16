from lab1.statistical import DataStats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def find_best_model_trend(x, y, title_x, title_y, max_degree):
    max_r2 = 0
    best_model = None
    datastats_old = None

    for degree in range(1, max_degree + 1):
        model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
        datastats = DataStats(x, y,
                                  title_x=title_x, title_y=title_y,
                                  model=model, model_name=f'Polynomial regression (degree={degree})')
        datastats.statistics(print_stats=False)
        dict_vals = datastats.plot_trend(plot_need=False, stats_r2_print=False)
        if dict_vals['R2'] > max_r2:
            #print(f'New best model: {datastats.model_name}, R^2: {dict_vals["R2"]}')
            max_r2 = dict_vals['R2']
            best_model = model
            datastats_old = datastats

    return best_model, max_r2, datastats_old


def results_trend(x, y, title_x, title_y, max_degree=30):
    best_model, max_r2, datastats_old = find_best_model_trend(x, y, title_x, title_y, max_degree)
    print(f'Найкраща модель: {datastats_old.model_name}, R^2: {max_r2}')
    datastats = DataStats(x, y, title_x=title_x, title_y=title_y, model=best_model, model_name=datastats_old.model_name)
    datastats.statistics(print_stats=True)
    datastats.plot_trend()

    return datastats