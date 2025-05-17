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


import plotly.express as px
df = pd.read_excel('./data/dou_data_clear.xlsx')
df_cube = df.groupby(['job_title', 'company_name', 'location']).size().reset_index(name='count')
fig = px.scatter_3d(df_cube,
                    x='job_title',
                    y='company_name',
                    z='location',
                    color='count',
                    title='3D Scatter Plot of Job Titles, Companies, and Locations',
                    labels={'job_title': 'Job Title', 'company_name': 'Company Name', 'location': 'Location'},
                    hover_name='count',
                    size_max=10,
                    color_continuous_scale=px.colors.sequential.Viridis)
fig.update_traces(marker=dict(size=5))
fig.show()


