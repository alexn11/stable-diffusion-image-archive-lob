# Copyright 2024 Alexandre De Zotti. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import os
import requests

from dotenv import load_dotenv
import streamlit as st


load_dotenv('app.env')
api_url = os.environ['API_URL']

def request_api(method: str, path: str, data: dict | None = None, fetching=False):
    print(f'sending query to "{path}"')
    url = os.path.join(api_url, path)
    if(fetching):
        image_load_state.text('fetching image...')
    try:
        if(method == 'get'):
            response = requests.get(url=url)
        else:
            response = requests.post(url=url, json=data)
    except Exception:
        image_load_state.empty()
        raise
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

def search_image(prompt):
    response_data = request_api('post', 'image/by-prompt', data={'prompt': prompt, }, fetching=True)
    #print(response_data)
    key = response_data['key']
    image = decode_image(response_data)
    return key, image

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

st.title('Stable Diffusion Image Archive')

   
image_load_state = st.empty()
image_place_holder = st.empty()

if 'key_text' not in st.session_state:
    st.session_state.key_text = ''

if 'prev_prompt' not in st.session_state:
    st.session_state.prev_prompt = None


# https://stackoverflow.com/questions/69492406/streamlit-how-to-display-buttons-in-a-single-line
col1, col2, col3, col4 = st.columns([1,1,1,1])
with col1:
    st.button('prev', key='btn_prev')
with col2:
    st.button('next', key='btn_next')
with col3:
    st.button('random', key='btn_random')
with col4:
    st.button('search again', key='btn_search_again')

st.code(st.session_state.key_text)
prompt = st.text_input('prompt search', key='prompt_input')


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
if(st.session_state.get('btn_search_again')):
    new_key, image = search_image(prompt)
    update_image(new_key, image)


if(st.session_state.get('prompt_input')):
    if(st.session_state.prev_prompt != prompt):
        st.session_state.prev_prompt = prompt
        new_key, image = search_image(prompt)
        update_image(new_key, image)

