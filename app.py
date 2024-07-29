import base64
import os
import requests

from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import numpy as np


load_dotenv('app.env')
api_url = os.environ['API_URL']

def request_api(method: str, path: str, data: dict | None = None, fetching=False):
    url = os.path.join(api_url, path)
    if(fetching):
        image_load_state.text('fetching image...')
    if(method == 'get'):
        response = requests.get(url=url)
    else:
        response = requests.post(url=url, json=data)
    response_data = response.json()
    if(fetching):
        image_load_state.empty()
    return response_data

def decode_image(response_data: str):
    return base64.b64decode(response_data['image'])

def get_random_key():
    response_data = request_api('get', 'key/random')
    return response_data.get('key', '')

def get_image(key):
    response_data = request_api('post', 'image', data={ 'key': key, }, fetching=True)
    return decode_image(response_data)

def get_next_image(key):
    response_data = request_api('post', 'image/next', data={ 'key': key, }, fetching=True)
    return response_data['key'], decode_image(response_data)

def get_prev_image(key):
    response_data = request_api('post', 'image/prev', data={ 'key': key, }, fetching=True)
    return response_data['key'], decode_image(response_data)

def update_image(key, image):
    image_place_holder.image(image,)
    st.session_state.key_text = key

def get_key():
    key = st.session_state.get('key_text')
    if(key == ''):
        key = get_random_key()
    return key

st.title('Stable diffusion babelia')

image_load_state = st.empty()
image_place_holder = st.empty()

if 'key_text' not in st.session_state:
    st.session_state.key_text = ''



# https://stackoverflow.com/questions/69492406/streamlit-how-to-display-buttons-in-a-single-line
col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.button('prev', key='btn_prev')
with col2:
    st.button('next', key='btn_next')
with col3:
    st.button('random', key='btn_random')

if(st.session_state.get('btn_prev')):
    new_key, image = get_prev_image(get_key())
    update_image(new_key, image)
if(st.session_state.get('btn_next')):
    new_key, image = get_next_image(get_key())
    update_image(new_key, image)
if(st.session_state.get('btn_random')):
    new_key = get_random_key()
    image = get_image(new_key)
    update_image(new_key, image)
if(st.session_state.get('prompt_input')):
    st.text('PROMPT INPUT')

st.text(st.session_state.key_text)

st.text_input('prompt', key='prompt_input')

