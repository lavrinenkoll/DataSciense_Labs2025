import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud


def generate_wordcloud(array: list, title: str):
    array = [str(x) for x in array if pd.notna(x)]

    text = ' '.join(array)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()


def generate_wordcloud_from_df(data: pd.DataFrame, title: str):
    job_titles = data['job_title'].tolist()
    level = data['level'].tolist()
    locations = data['location'].tolist()
    companies = data['company_name'].tolist()

    skills = data['skills'].tolist()
    skills = [skill.split(',') for skill in skills]
    offers = data['offers'].tolist()
    offers = [offer.split(',') for offer in offers]

    # Flatten the list of lists
    skills = [item for sublist in skills for item in sublist]
    offers = [item for sublist in offers for item in sublist]

    #drop "none"
    skills = [skill for skill in skills if skill != 'none']
    offers = [offer for offer in offers if offer != 'none']

    # Generate word clouds
    generate_wordcloud(job_titles, f'Job Titles | {title}')
    generate_wordcloud(level, f'Levels | {title}')
    generate_wordcloud(locations, f'Locations | {title}')
    generate_wordcloud(companies, f'Companies | {title}')
    generate_wordcloud(skills, f'Skills | {title}')
    generate_wordcloud(offers, f'Company Offers | {title}')


def clean_title(title):
    if pd.isna(title):
        return ''
    title = title.lower().strip()
    title = re.sub(r'[^\w\s\-/]', '', title)
    return title


def compare_column(df1: pd.DataFrame, df2: pd.DataFrame, column: str, clean_func, title: str, save_path: str, top_n=10):
    df1[f'{column}_clean'] = df1[column].dropna().apply(clean_func)
    df2[f'{column}_clean'] = df2[column].dropna().apply(clean_func)

    counts1 = df1[f'{column}_clean'].value_counts()
    counts2 = df2[f'{column}_clean'].value_counts()

    common_items = set(counts1.index).intersection(set(counts2.index))
    common_counts_sum = {item: counts1.get(item, 0) + counts2.get(item, 0) for item in common_items}
    top_common_items = sorted(common_counts_sum, key=common_counts_sum.get, reverse=True)[:top_n]

    values1 = [counts1.get(item, 0) for item in top_common_items]
    values2 = [counts2.get(item, 0) for item in top_common_items]

    x = np.arange(len(top_common_items))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, values1, width, label='DOU')
    plt.bar(x + width / 2, values2, width, label='LinkedIn')
    plt.xticks(x, top_common_items, rotation=45, ha='right')
    plt.xlabel(title)
    plt.ylabel('Counts')
    plt.title(f'Top Common {title} Comparison')
    plt.legend()
    plt.tight_layout()
    plt.show()


def compare_dfs(df1: pd.DataFrame, df2: pd.DataFrame):
    compare_column(df1, df2, 'job_title', clean_title, 'Job Titles', 'top_common_job_titles_comparison.png')
    compare_column(df1, df2, 'level', clean_title, 'Levels', 'top_common_levels_comparison.png')
    compare_column(df1, df2, 'location', clean_title, 'Locations', 'top_common_locations_comparison.png')
    compare_column(df1, df2, 'company_name', clean_title, 'Companies', 'top_common_companies_comparison.png')


def generate_olap_cube(df: pd.DataFrame, index_columns: list, value_column: str, specific_location: str,
                          title: str, top_n: int = 30):
    def count_specific_value(x, target):
        return (x == target.lower()).sum()

    # Очистка назв
    df[index_columns[0]] = df[index_columns[0]].apply(clean_title)
    df[index_columns[1]] = df[index_columns[1]].apply(clean_title)
    df[value_column] = df[value_column].apply(clean_title)

    # Top N
    top_jobs = df[index_columns[0]].value_counts().head(top_n).index
    top_companies = df[index_columns[1]].value_counts().head(top_n).index

    # Фільтрація
    df_filtered = df[df[index_columns[0]].isin(top_jobs) & df[index_columns[1]].isin(top_companies)]

    # Зведена таблиця
    pivot_table = pd.pivot_table(
        df_filtered,
        index=index_columns[0],
        columns=index_columns[1],
        values=value_column,
        aggfunc=lambda x: count_specific_value(x, specific_location),
        fill_value=0
    )
    pivot_table = pivot_table.loc[:, (pivot_table.sum(axis=0) > 0)]
    pivot_table = pivot_table.loc[(pivot_table.sum(axis=1) > 0), :]

    # Підготовка даних для 3D графіку
    x_labels = list(pivot_table.index)
    y_labels = list(pivot_table.columns)
    x_len = len(x_labels)
    y_len = len(y_labels)

    xpos, ypos = np.meshgrid(np.arange(x_len), np.arange(y_len), indexing="ij")
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)

    # Висоти стовпчиків
    dz = pivot_table.values.flatten()

    # Розміри
    dx = dy = 0.4

    # Колір на основі значень
    from matplotlib import cm
    colors = cm.viridis(dz / dz.max())

    # Побудова 3D-графіку
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, zsort='average', alpha=0.9)

    ax.set_xlabel(index_columns[0])
    ax.set_ylabel(index_columns[1])
    ax.set_zlabel(f'"{specific_location}" in {value_column}')

    # Індекси на осях
    ax.set_xticks(np.arange(x_len) + dx / 2)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_yticks(np.arange(y_len) + dy / 2)
    ax.set_yticklabels(y_labels, rotation=-35, ha='left')
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='z', labelsize=10)

    plt.title(title)
    plt.show()




