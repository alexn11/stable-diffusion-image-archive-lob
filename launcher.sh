#!/bin/bash

if [ ! -d .venv ]; then
    python -m venv .venv;
    source .venv/bin/activate;
    pip install -r requirements.txt;
fi;

fastapi api.py &
streamlit run app.py &
