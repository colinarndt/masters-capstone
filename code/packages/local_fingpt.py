import requests
import time
import backoff

@backoff.on_exception(
    backoff.expo,
    (Exception),  
    max_tries=5
)
def query_local_fingpt(headline, company, api_url="http://localhost:8080/completion"):
    """
    Query locally running Llama3-8B model with FinGPT LoRA adapter 
    """
    request = {
        "prompt": f"""
        <|start_header_id|>user<|end_header_id|>
        
            You are analyzing this headline about {company} as an experienced stock analyst. 
            Your goal is to give recommendations that lead to a profitable trade.

            Score the headline's market impact, considering:
            - Most news has minimal real impact (use 0)
            - Consider market expectations vs reality
            - A negative sentiment should yield a negative score

            Use these scores only: -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1

            Headline: {headline}

            Respond only in this format:
            SCORE: [number]
            REASON: [one sentence explanation]
        
        <|start_header_id|>assistant<|end_header_id|>
        """,
        "n_predict": 32,
        "temperature": 0,
        "stop": ["Human:", "Assistant:", "\n\n\n"]
    }
    
    response = requests.post(api_url, json=request)
    return response.json()['content']