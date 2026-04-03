# Turing Test Judge Benchmark — Evaluation Script
================================================
Given a dataset of paired dialogues (A and B), predict which is the human-human dialogue.

## SETUP
-----
1. Install core dependencies:
       pip install pandas tqdm datasets

2. Install whatever library your model needs (see examples below).

3. Fill in the `predict()` function in `predict.py` with your model.

4. Configure necessary changes in `config.py`. Here you can adjust the prompt, set multithreading config, and handle API call delay.

4. Run:
       ### Default
       python run_eval.py

       ### Save output to a custom path
       python run_eval.py --output my_predictions.csv

       ### Add a delay between API calls (seconds, useful for rate limits)
       python run_eval.py --delay 0.5

## OUTPUT FORMAT
-------------
A single-column CSV:  who_is_human  ∈  {"A", "B"}

## MULTITHREADING
--------------
Set USE_THREADS = True below to enable parallel inference.
Set N_THREADS to control the number of worker threads.
Recommended for API-based models (OpenAI, Groq, Together, etc.).
NOT recommended for local models (transformers, Ollama) — use N_THREADS = 1.0