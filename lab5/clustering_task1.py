from lab1.parser import Parser
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def calculate_price_clusters(price_change, find_optimal = True, max_clusters=7):
    df = pd.DataFrame(price_change)
    df['date'] = pd.to_datetime(df['date'])
    df['date_ordinal'] = df['date'].map(pd.Timestamp.toordinal)
    X = df[['date_ordinal', 'avg']].values

    if find_optimal:
        best_silhouette_score = -1
        optimal_num_clusters = 2
        print('-' * 50)
        for num_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            silhouette_avg = silhouette_score(X, cluster_labels)

            if silhouette_avg > best_silhouette_score:
                best_silhouette_score = silhouette_avg
                optimal_num_clusters = num_clusters
            print(f'Кількість кластерів: {num_clusters}, Silhouette Score: {silhouette_avg:.4f}')
        print('-' * 50)
        print(f'Оптимальна кількість кластерів: {optimal_num_clusters} (Silhouette Score: {best_silhouette_score:.4f})')
    else:
        optimal_num_clusters = max_clusters

    kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    clusters = defaultdict(list)
    for i, label in enumerate(cluster_labels):
        clusters[label].append(X[i])

    cluster_ranges = []
    for cluster_points in clusters.values():
        cluster_points = np.array(cluster_points)
        min_date = int(np.min(cluster_points[:, 0]))
        max_date = int(np.max(cluster_points[:, 0]))
        min_price = np.min(cluster_points[:, 1])
        max_price = np.max(cluster_points[:, 1])
        cluster_ranges.append(((min_date, max_date), (min_price, max_price)))

    cluster_ranges = sorted(cluster_ranges, key=lambda x: x[0][0])

    # Візуалізація
    plt.figure(figsize=(12, 6))
    for i, ((min_d, max_d), (min_p, max_p)) in enumerate(cluster_ranges):
        plt.fill_betweenx(
            [min_p, max_p],
            min_d,
            max_d,
            alpha=0.4,
            label=f'Кластер {i + 1}',
            linestyle='-'
        )

    plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', s=50)
    plt.xlabel('Дата (ordinal)')
    plt.ylabel('Середня ціна')
    plt.title('Кластеризація середніх цін по датах')
    plt.grid()
    plt.legend()
    plt.show()

    return cluster_ranges


def plot_graph(x, y, title_x, title_y):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.title(f'Графік залежності {title_y.upper()} від {title_x.upper()}')
    plt.xlabel(title_x)
    plt.ylabel(title_y)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    url = input('Введіть URL для парсингу (сторінка товару на ekatalog): ')
    if not url:
        print('URL не введено. Використовується URL за замовчуванням.')
        url = 'https://ek.ua/ua/CANON-50MM-F-1-4-EF-USM.htm'
    #url = 'https://ek.ua/ua/APPLE-IPHONE-16-PRO-MAX-256GB.htm'
    print('-' * 50)

    parser = Parser(url)
    price_change, number_of_shops = parser.parse_table(save_to_files=False)
    plot_graph(price_change['date'], price_change['avg'], 'Дата', 'Середня ціна')

    optimal_clusters_needed = input('Вам потрібно знайти оптимальну кількість кластерів? (y/n): ')
    if optimal_clusters_needed.lower() == 'y':
        find_optimal = True
        print('Обрано знаходити оптимальну кількість кластерів.')
        max_clusters = input('Введіть максимальну кількість кластерів (за замовчуванням 15): ')
        if not max_clusters:
            print('Максимальна кількість кластерів не введена. Використовується значення за замовчуванням (15).')
            max_clusters = 15
        else:
            max_clusters = int(max_clusters)
        clusters = calculate_price_clusters(price_change, find_optimal=find_optimal, max_clusters=max_clusters)
    else:
        print('Обрано не знаходити оптимальну кількість кластерів.')
        find_optimal = False
        max_clusters = input('Введіть кількість кластерів: ')
        if not max_clusters:
            print('Кількість кластерів не введена. Використовується значення за замовчуванням (7).')
            max_clusters = 7
        else:
            max_clusters = int(max_clusters)
        clusters = calculate_price_clusters(price_change, find_optimal=find_optimal, max_clusters=max_clusters)
    print('-' * 50)

