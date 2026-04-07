## Overview
This codebase includes the core Jupyter notebooks used to generate the results in this project.

## Project Structure
Data_preparation.ipynb handles the data filtering, denoising and stratified sampling. 

A headline_analysis notebook was created for each model. Separating the notebooks allowed viewing a summary of performance results within each notebook.

Helper functions are in the 'packages' folder.

Some files could not be uploaded to GitHub due to size. To run this code yourself, you will need to:
1. Create a folder called 'source' and load 'analyst_ratings_processed.csv' into it from the below Kaggle link.
2. Create a folder called 'output2' then run the Data_preparation notebook. This will output the final filtered & sampled dataset, ready for use in the headline_analysis notebooks.

The code used to compute final metrics and visualizations is not included as it was boilerplate - it uses metric calculations and visualization libraries which are standard in data science.

## Dataset
The dataset was obtained from Kaggle:
https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests

## Local Llama models and llama.cpp
Llama.cpp can be installed on Mac with the brew package manager:

`brew install llama.cpp`

After downloading a model to the 'models' folder in your llama.cpp directory, it can be served with the following command:

`./build/bin/llama-server -m models/<model name> -c 2048 --host 0.0.0.0 --port 8080`

To run the model with the QLoRA adapter:

`./build/bin/llama-server -m models/llama-3-8b-instruct.Q8_0.gguf --lora models/fingpt-mt_llama3-8b_lora-q8_0.gguf -c 2048 --host 0.0.0.0 --port 8080`

Local Llama models used in this project are linked below. 

Since this was a comparative analysis, seeking to bring greater understanding and transparency to the field, rather than generate new models or results, I used existing models and QLoRA adaptors rather than fine-tuning my own. 

I had hoped to validate the results of FinLlama and FinLLaMa as well, but unfortunately, at the time of this project they were not available or not suitable to the project.

Llama3-8B_Q8
https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct-Q8_0.gguf

Llama3-8B_Q6
https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct-Q6_K.gguf

Llama3-8B_Q4
https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf

Llama3-8B_Q8 QLoRA
https://huggingface.co/second-state/FinGPT-MT-Llama-3-8B-LoRA-GGUF



