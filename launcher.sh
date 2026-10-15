#!/bin/bash

if [ ! -d .venv ]; then
    uv .venv;
    source .venv/bin/activate;
    uv pip install -r requirements.txt;
else
    source .venv/bin/activate;
fi;

fastapi run api.py &
streamlit run app.py &
