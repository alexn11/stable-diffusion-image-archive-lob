import base64
import io

from fastapi import FastAPI
from pydantic import BaseModel, constr, validator

from model_constants import key_length
from key_strings import base64_characters

from AppConfig import AppConfig
from ImageCache import ImageCache
from ImageFinder import ImageFinder

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

def get_image(key: str | None, prompt: str | None = None):
    if((key is None) and (prompt is not None)):
        key = image_finder.find_a_key(prompt)
    if(key is not None):
        key = validate_key(key)
        image = image_cache.get(key, create_func=image_finder.find, creat_func_args={'key': key})
        image_byte_stream = io.BytesIO()
        image.save(image_byte_stream, format="JPEG")
        encoded_image_base64 = base64.b64encode(image_byte_stream.getvalue()).decode("utf-8")
    if(prompt is None):
        return encoded_image_base64
    return encoded_image_base64, key

app_config = AppConfig()
app = FastAPI()
image_cache = ImageCache(cache_folder_path=app_config.image_cache_folder)
image_finder = ImageFinder(app_config.image_finder_config_dict)

@app.post('/image')
def key_to_image(key: str):
    encoded_image_base64 = get_image(key=key)
    return {
        'key': key,
        'image': encoded_image_base64,
    }

@app.get('/image/by-prompt')
def promt_to_image(prompt: str):
    encoded_image_base6, image_key = get_image(prompt=prompt)
    return {
        'key': image_key,
        'image': encoded_image_base6,
    }
