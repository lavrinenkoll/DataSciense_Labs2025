# Розробити програмний скрипт, що реалізує пошук усіх пропозицій ринку праці на посаду
# аналітика даних. Передбачити OLAP – візуалізацію результатів.
import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options


def get_urls_dou(search_str):
    # parse data from dou
    dou_url = f'https://jobs.dou.ua/vacancies/?search={search_str.replace(" ", "+")}'

    # chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # run in headless mode
    chrome_options.add_argument('--no-sandbox')  # disable sandboxing

    # create a new instance of the Chrome driver
    driver = webdriver.Chrome(options=chrome_options)

    # open the URL
    driver.get(dou_url)
    # wait for the page to load
    driver.implicitly_wait(10)

    # find the button
    while True:
        try:
            more_btn = driver.find_element(By.XPATH, '//div[@class="more-btn"]/a')
            more_btn.click()
        except Exception as e:
            #print(e)

            # get the page source
            requests_html = driver.page_source

            # close the driver
            driver.quit()
            break

    soup = BeautifulSoup(requests_html, 'html.parser')
    job_cards = soup.find_all('li', class_='l-vacancy')

    # extract urls
    job_urls = []
    for card in job_cards:
        try:
            url = card.find('a', class_='vt')['href']
            job_urls.append(url)
        except Exception as e:
            #print(e)
            continue
    return job_urls


def grab_dou_vacancy(url):
    headers = {
        'User-Agent': 'Mozilla/5.0'
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    data = {}

    # Заголовок
    title = soup.find('h1', class_='g-h2')
    data['job_title'] = title.text.strip() if title else None

    # Компанія
    company = soup.select_one('div.info div.l-n a')
    data['company_name'] = company.text.strip() if company else None

    # Зарплата
    salary = soup.find('span', class_='salary')
    data['salary'] = salary.text.strip() if salary else None

    # Дата
    date = soup.find('div', class_='date')
    data['date'] = date.text.strip() if date else None

    # Локація
    location = soup.select_one('span.place')
    data['location'] = location.text.strip() if location else None

    # Опис (текст)
    description_block = soup.find('div', class_='b-typo vacancy-section')
    data['description_text'] = description_block.get_text(separator="\n").strip() if description_block else None

    return data


def get_urls_linkedin(search_str):
    # parse data from linkedin
    linkedin_url = f'https://www.linkedin.com/jobs/search?keywords={search_str.replace(" ", "%20")}&location=Ukraine&geoId=102264497&position=1&pageNum=0'

    # chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # run in headless mode
    chrome_options.add_argument('--no-sandbox')  # disable sandboxing

    # create a new instance of the Chrome driver
    driver = webdriver.Chrome(options=chrome_options)

    # open the URL
    driver.get(linkedin_url)
    # wait for the page to load
    driver.implicitly_wait(10)

    # scroll down to load more jobs - while the button is not found infinite-scroller__show-more-button infinite-scroller__show-more-button--visible
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        if driver.find_elements(By.XPATH, '//button[@class="infinite-scroller__show-more-button infinite-scroller__show-more-button--visible"]'):
            break

    # get the page source
    requests_html = driver.page_source
    # close the driver
    driver.quit()

    soup = BeautifulSoup(requests_html, 'html.parser')
    job_cards_table = soup.find_all('ul', class_='jobs-search__results-list')

    job_cards = job_cards_table[0].find_all('li')

    # extract urls
    job_urls = []
    for card in job_cards:
        try:
            url = card.find('a', class_='base-card__full-link')['href']
            job_urls.append(url)
        except Exception as e:
            #print(e)
            continue

    return job_urls


def grab_linkedin_vacancy(url):
    headers = {
        'User-Agent': 'Mozilla/5.0'
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    data = {}

    # Заголовок
    title = soup.find('h1', class_='top-card-layout__title')
    data['job_title'] = title.text.strip() if title else None

    # Компанія
    company = soup.find('a', class_='topcard__org-name-link')
    data['company_name'] = company.text.strip() if company else None

    # Зарплата
    # not available on linkedin

    # Дата
    date = soup.find('span', class_='posted-time-ago__text')
    data['date'] = date.text.strip() if date else None

    # Локація
    location = soup.find('span', class_='topcard__flavor topcard__flavor--bullet')
    data['location'] = location.text.strip() if location else None

    # Опис (текст)
    description_block = soup.find('div', class_='show-more-less-html__markup')
    data['description_text'] = description_block.get_text(separator="\n").strip() if description_block else None

    return data


def grab_and_save(get_urls_func, grab_func, search_str, filename, site_name):
    urls = get_urls_func(search_str)
    print(f'Found URLs from {site_name.upper()}:', len(urls))
    data = pd.DataFrame(columns=['job_title', 'company_name', 'salary', 'date', 'location', 'description_text'])
    for url in urls:
        data_dict = grab_func(url)
        if data_dict['job_title'] is None:
            continue
        data = pd.concat([data, pd.DataFrame([data_dict])], ignore_index=True)
    data.to_excel(filename, index=False)

    return data


# # parse data from dou and linkedin
# search_str = 'data analyst'
# data_dou = grab_and_save(get_urls_dou, grab_dou_vacancy, search_str, './data/dou_data.xlsx', 'dou')
# data_linkedin = grab_and_save(get_urls_linkedin, grab_linkedin_vacancy, search_str, './data/linkedin_data.xlsx', 'linkedin')

# Load data from excel files
data_dou = pd.read_excel('./data/dou_data.xlsx')
data_linkedin = pd.read_excel('./data/linkedin_data.xlsx')






