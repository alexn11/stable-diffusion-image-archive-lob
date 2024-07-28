
from fastapi import FastAPI
from pydantic import BaseModel, constr, validator

from model_constants import key_length
from key_strings import base64_characters

def validate_key(key):
    key = ''.join(list(filter(lambda c: c in base64_characters, key)))
    if(len(key) != key_length):
        this_len = len(key)
        if(this_len > key_length):
            key = key[:key_length]
        else:
            padding_len = key_length - this_len
            key += padding_len * 'A'
    return key
# key: str = constr(pattern=rf'[{base64_characters}]{{{2}}}$')

app = FastAPI()


@app.post('/image/')
def key_to_image(key: str):
    key = validate_key(key)
    return {
        'key': key,
        'image': [],
    }

@app.get('/image/by-prompt/{prompt}')
def promt_to_image(prompt: str):
    return {
        'key': '',
        'image': [],
    }
