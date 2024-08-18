# Introduction

This is an attempt for a stable diffusion version of [babel image archives](http://babelia.libraryofbabel.info/about.html).

The idea is to replace image locations with keys encoding the image generation process in a fully deterministic way.
Like with the [text version using GPT-2](https://github.com/alexn11/gpt-2-library-of-babel), I was hoping to reduce the size of the location (✅️) and produce less random images (❌️).

# How to run

1. Create an environment with the requirements installed by running the following in the root folder of the project:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2. Launch the API server: 
```bash
fastapi api.py
```
3. Wait that this is ready (should show: "INFO:     Application startup complete.")
4. Launch the app:
```bash
streamlit run app.py
```
5. This should automatically open the app on a tab of a web browser.

# Desciption of the main app

![example of prompt look-up](images/prompt-lob.png)


In the app you can see:
- The image it found (after any request).
- A "random" button: generate a random key and its associated image.
- "prev" and "next" buttons: retrieve the image associated with either the preceding or subsequent key (keys can theoretically be listed sequentially).
- A "search again" button: search another image corresponding to the prompt given in the "prompt search" input.
- Below the buttons, a text area showing the key (with a "copy" button on its right side).
- Below the the key, a text input to search an image with a specific content: this will find a new key for an image that matches the prompt (or at least attempts to).


# Technical details

The keys encode different elements of a stable diffusion generation process used to ensure that the result is deterministic:
- the prompt embedding vector (887010 bits)
- a seed image (249600 bits)
- the number of inference steps (6 bits)

The keys are base 64 encoded binary strings of total length 189436.

The bits corresponding to the embedding vector and seed image are converted into 16 bits floats. The exponents are encoded within the key with 4 bits instead of 5 with a specific bias for each. This ensures that the length of the key is minimal and the values generated are within the expected model data distribution.

When no prompt is given, the prompt embedding vector is generated randomly within some region close to the region where real prompt embeddings lie.

## Other details

- Keys are ordered in some arbitrary way, the image depends on the key in a continuous way with respect to this order.
- References to image locations/keys  are much shorter than in the original: around 1,000,000 digits for an image on [the Library Of Babel Image Archive](https://babelia.libraryofbabel.info) vs ~190,000 on this version.
- Despite ensuring full determinism for a single system, the images generated might depend on the system it runs on, resulting in different libraries on different systems.
- It's not possible to search for a specific image but it's possible to try and search for an image with a specific description using the prompt.

![example of randomly generated image](images/random-key.png)

# Things that don't work

- Even though it's using stable diffusion, a random key usually produces a boring image.
- Unfortunately there is no guarantee that several keys wouldn't produce the same image.
- Similarly, there is no proof that any particular image can be reached by choosing the right key.
- Descriptions provided in the prompt are sometimes disregarded.
- The negative prompt embeddings are not generated as they should.



# Credits

The stable diffusion generation loop comes from Hugging Face's *diffusers* code, from [their stable diffusion pipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py) with some modifications.






















