import requests, openai, wikipediaapi, os
from bs4 import BeautifulSoup

self_api_key = os.environ.get('OPENAI_API_KEY')
BASE_URL = os.environ.get('BASE_URL')

if BASE_URL:
    client = openai.OpenAI(
        api_key=self_api_key,
        base_url=BASE_URL,
    )
else:
    client = openai.OpenAI(
        api_key=self_api_key
    )

def get_baidu_baike_content(keyword):
    url = f'https://baike.baidu.com/item/{keyword}'
    response = requests.get(url)

    soup = BeautifulSoup(response.content, 'html.parser')
    main_content = soup.contents[-1].contents[0].contents[4].attrs['content']
    return main_content

def get_wiki_content(keyword):
    wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)', 'en')
    search_topic = keyword
    page_py = wiki_wiki.page(search_topic)
    if page_py.exists():
        print("Page - Title:", page_py.title)
        print("Page - Summary:", page_py.summary)
    else:
        print("Page not found.")
    return page_py.summary

def modal_trans(task_dsp):
    try:
        task_in ="'" + task_dsp + \
               "'Just give me the most important keyword about this sentence without explaining it and your answer should be only one keyword."
        messages = [{"role": "user", "content": task_in}]
        response = client.chat.completions.create(messages=messages,
        model="gpt-3.5-turbo-16k",
        temperature=0.2,
        top_p=1.0,
        n=1,
        stream=False,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        logit_bias={})
        response_text = response.choices[0].message.content
        spider_content = get_wiki_content(response_text)
        task_in = "'" + spider_content + \
               "',Summarize this paragraph and return the key information."
        messages = [{"role": "user", "content": task_in}]
        response = client.chat.completions.create(messages=messages,
        model="gpt-3.5-turbo-16k",
        temperature=0.2,
        top_p=1.0,
        n=1,
        stream=False,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        logit_bias={})
        result = response.choices[0].message.content
        print("web spider content:", result)
    except:
        result = ''
        print("the content is none")
    return result