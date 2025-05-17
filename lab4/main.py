import pandas as pd
from lab4.parsing_process import grab_and_save, get_urls_dou, grab_dou_vacancy, get_urls_linkedin, grab_linkedin_vacancy
from lab4.clear_data import clear_data_dou, clear_data_linkedin
from lab4.data_analytics import generate_wordcloud_from_df, compare_dfs, generate_olap_cube

if __name__ == '__main__':
    #search_str = 'data analyst'
    search_str = input('Enter search string: ').lower().strip()
    PARSE_NEED = input('Do you want to parse data? (y/n): ').strip().lower() == 'y'
    CLEAR_NEED = input('Do you want to clear data? (y/n): ').strip().lower() == 'y'
    print('-' * 50)

    if PARSE_NEED:
        # parse data from dou and linkedin
        print('Parsing data...')
        print('Parsing data from DOU...')
        data_dou = grab_and_save(get_urls_dou, grab_dou_vacancy, search_str, './data/dou_data.xlsx', 'dou')
        print('Parsing data from LinkedIn...')
        data_linkedin = grab_and_save(get_urls_linkedin, grab_linkedin_vacancy, search_str, './data/linkedin_data.xlsx', 'linkedin')
    else:
        # Load data from excel files
        print('Loading data from excel files...')
        data_dou = pd.read_excel('./data/dou_data.xlsx')
        data_linkedin = pd.read_excel('./data/linkedin_data.xlsx')

    if CLEAR_NEED:
        # Clear data
        print('Clearing data...')
        data_dou_clear = clear_data_dou(data_dou, search_str, call_api_location=True, call_api_description=True)
        data_linkedin_clear = clear_data_linkedin(data_linkedin, search_str, call_api_location=True, call_api_description=True)
    else:
        # load cleared data if already cleared
        print('Loading cleared data from excel files...')
        data_dou_clear = pd.read_excel('./data/dou_data_clear.xlsx')
        data_linkedin_clear = pd.read_excel('./data/linkedin_data_clear.xlsx')

    # analyze data
    print('Analyzing data...')
    print('Generating word clouds...')
    merge_data = pd.concat([data_dou_clear, data_linkedin_clear], ignore_index=True)
    generate_wordcloud_from_df(merge_data, title='Data: DOU + LinkedIn')
    print('Comparing data...')
    compare_dfs(data_dou_clear, data_linkedin_clear)
    print('OLAP analysis...')
    generate_olap_cube(merge_data, ['job_title', 'location'], 'level',
                       'senior',
                       title='OLAP Analysis: count of senior positions by job title and location')
    generate_olap_cube(merge_data, ['job_title', 'company_name'], 'location',
                       'remote',
                       title='OLAP Analysis: count of remote positions by job title and company name')
    generate_olap_cube(merge_data, ['level', 'location'], 'job_title',
                          'data analyst',
                          title='OLAP Analysis: count of data analyst positions by level and location')








