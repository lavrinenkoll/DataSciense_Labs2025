import re
import pandas as pd
from deep_translator import GoogleTranslator
from lab4.ai_model_process import get_response_api
import logging


def clear_and_reformat_job_titles(data: pd.DataFrame, search_str) -> pd.DataFrame:
    words = search_str.split()
    data = data[data['job_title'].str.contains('|'.join(words), case=False)].copy()

    data['level'] = data['job_title'].str.extract(r'(?i)(junior|middle|senior|intern|trainee|lead)')
    cols = data.columns.tolist()
    cols.insert(1, cols.pop(cols.index('level')))
    data = data[cols]

    data['job_title'] = data['job_title'].str.replace(r'(?i)(junior|middle|senior|intern|trainee|lead)', ' ', regex=True)
    data['job_title'] = data['job_title'].str.replace(r'\s+', ' ', regex=True)
    data['job_title'] = data['job_title'].str.strip()

    data['job_title'] = data['job_title'].str.replace(r'\(.*?\)', '', regex=True)

    def is_latin(text: str) -> bool:
        return bool(re.fullmatch(r'[a-zA-Z0-9\s\.\,\-\(\)\/\+]+', text))

    def translate_if_needed(text: str) -> str:
        if pd.isna(text) or is_latin(text):
            return text
        try:
            return GoogleTranslator(source='auto', target='en').translate(text)
        except Exception as e:
            logging.error(f"Translation error: {e}")
            return text

    data['job_title'] = data['job_title'].apply(translate_if_needed)

    return data


def process_location(data: pd.DataFrame, call_api = False) -> pd.DataFrame:
    system_prompt = '''
    Identify the job location using the correct format. I will force you instead of the field, and you turn my word or words through the space. 
    Rules: 
    1) The result of the English language. 
    2) If indicated only for Ukraine - write remote. 
    3) If it is remote or remote - write remote. 
    4) If there are other countries, write them. 
    5) If there is a place - write with a space. 
    6) If there is one place - write yogo. 
    7) If it is “beyond the cordon”, then it is a Ukrainian place - hence it is no longer a Ukrainian place. 
    8) No more confirmation, nothing funny. 
    
    Check out this location:'''

    for index, row in data.iterrows():
        location = row['location']
        if pd.isna(location):
            continue

        if len(location.split()) == 1 and location.lower() == 'ukraine':
            data.at[index, 'location'] = 'remote'
        elif location.lower().find('remote') != -1 or location.lower().find('віддалено') != -1:
            data.at[index, 'location'] = 'remote'
        elif len(location.split()) == 1:
            data.at[index, 'location'] = GoogleTranslator(source='auto', target='en').translate(location)
        elif call_api:
            prompt_current = GoogleTranslator(source='auto', target='en').translate(location)
            response = get_response_api(prompt_current, system_prompt=system_prompt)
            data.at[index, 'location'] = GoogleTranslator(source='auto', target='en').translate(response)

    return data


def process_description(data: pd.DataFrame) -> pd.DataFrame:
    system_prompt = '''
    You are an expert job description parser. Your task is to extract the key skills and company benefits even if they are not explicitly mentioned. Follow these instructions carefully:

    Format your response as follows:
    Line 1: Required skills/technologies (in English, comma-separated, lowercase if possible, key phrases only, no duplicates)
    Line 2: Company offers (in English, comma-separated, lowercase if possible, short phrases, no duplicates)

    Rules:
    1) If skills are not clearly mentioned but can be inferred from context (e.g. "you will build dashboards" → "data visualization"), list them anyway.
    2) If company benefits are not explicitly listed but can be inferred (e.g. "you will work with international teams" → "international environment"), include them.
    3) If nothing can be identified even indirectly, write "none" on that line.
    4) Do not add any explanation, comments, or extra formatting.
    5) Output only two lines, no bullets, no numbering, no labels.
    6) Be concise. Avoid adjectives or fluff. Only concrete items.
    '''

    skills_list = []
    offers_list = []

    for description in data['description_text'].fillna(''):
        if not description.strip():
            skills_list.append('none')
            offers_list.append('none')
            continue

        try:
            if re.fullmatch(r'[a-zA-Z0-9\s\.,\-()]+', description.strip()):
                description_en = description
            else:
                description_tail = description[-4000:] if len(description) > 4000 else description
                description_en = GoogleTranslator(source='auto', target='en').translate(description_tail)
        except Exception as e:
            logging.error(f"Translation error: {e}")
            description_en = description

        response = get_response_api(description_en, system_prompt=system_prompt)

        lines = response.strip().split('\n')
        if len(lines) >= 2:
            skills = lines[0].strip() or 'none'
            offers = lines[1].strip() or 'none'
        else:
            skills = 'none'
            offers = 'none'

        skills_list.append(skills)
        offers_list.append(offers)

    data['skills'] = skills_list
    data['offers'] = offers_list

    cols = [col for col in data.columns if col != 'description_text'] + ['description_text']
    data = data[cols]

    return data


def del_unuseful_columns(data: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = ['salary', 'date']
    data = data.drop(columns=columns_to_drop, errors='ignore')
    return data


def clear_data_dou(data_dou: pd.DataFrame, search_str, call_api_location = False, call_api_description = False) -> pd.DataFrame:
    data_dou = data_dou.dropna(subset=['job_title'])
    data_dou = clear_and_reformat_job_titles(data_dou, search_str)

    data_dou = del_unuseful_columns(data_dou)

    data_dou = process_location(data_dou, call_api_location)

    if call_api_description:
        data_dou = process_description(data_dou)

    data_dou.to_excel('./data/dou_data_clear.xlsx', index=False)
    return data_dou


def clear_data_linkedin(data_linkedin: pd.DataFrame, search_str, call_api_location = False, call_api_description = False) -> pd.DataFrame:
    data_linkedin = data_linkedin.dropna(subset=['job_title'])
    data_linkedin = clear_and_reformat_job_titles(data_linkedin, search_str)

    data_linkedin = del_unuseful_columns(data_linkedin)

    data_linkedin = process_location(data_linkedin, call_api_location)
    if call_api_description:
        data_linkedin = process_description(data_linkedin)

    data_linkedin.to_excel('./data/linkedin_data_clear.xlsx', index=False)
    return data_linkedin

