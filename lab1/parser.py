import requests
from bs4 import BeautifulSoup
import pandas as pd
import json


class Parser:
    '''
    A class to parse the data from the given URL.
    '''

    def __init__(self, url):
        '''
        Initialize the parser with the given URL.
        :param url: str
        '''
        self.url = url

    def fetch_data(self):
        '''
        Fetch the data from the given URL.
        :return: BeautifulSoup object of the fetched data or None if failed
        '''
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(self.url, headers=headers)

        if response.status_code == 200:
            return BeautifulSoup(response.text, "html.parser")
        return None

    @staticmethod
    def transform_ms_to_date(ms):
        '''
        Transform milliseconds to date.
        :param ms: int
        :return: str
        '''
        return pd.to_datetime(ms, unit='ms')

    def parse_table(self, save_to_files=False):
        '''
        Parse the data from the fetched data.
        :return: tuple of two DataFrames or None if failed
        '''
        data = self.fetch_data()
        if data is None:
            return None

        # find 2 of EKCharts.init
        scripts = data.find_all('script')
        tables = []
        for script in scripts:
            if 'EKCharts.init' in script.text:
                tables.append(script.text)
        tables = [tables.split('"data":')[1].split('}]')[0]+'}]' for tables in tables]

        # parse the tables
        data_json = json.loads(tables[0])
        price_change = pd.DataFrame(data_json)

        data_json = json.loads(tables[1])
        number_of_shops = pd.DataFrame(data_json)

        # transform the date
        price_change['date'] = price_change['date'].apply(self.transform_ms_to_date)
        number_of_shops['date'] = number_of_shops['date'].apply(self.transform_ms_to_date)

        #transform the values
        price_change['max'] = price_change['max'].apply(lambda x: float(x.split(' ')[0]))
        price_change['min'] = price_change['min'].apply(lambda x: float(x.split(' ')[0]))
        price_change['avg'] = price_change['avg'].apply(lambda x: float(x.split(' ')[0]))
        number_of_shops['value'] = number_of_shops['value'].apply(lambda x: int(x))

        # # add date_num column
        # price_change['date_num'] = price_change['date'].apply(lambda x: x.toordinal())
        # number_of_shops['date_num'] = number_of_shops['date'].apply(lambda x: x.toordinal())

        if save_to_files:
            name = self.url.split('/')[-1].split('.')[0]
            price_change.to_excel(f'./data/{name}_price_change.xlsx', index=False)
            number_of_shops.to_excel(f'./data/{name}_number_of_shops.xlsx', index=False)

        return price_change, number_of_shops


