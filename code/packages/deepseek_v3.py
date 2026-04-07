import time
import backoff
import requests

@backoff.on_exception(
    backoff.expo,
    (Exception),  
    max_tries=5
)
def query_deepseek(headline, company, api_key=None, model="deepseek-chat"):
    """
    Query DeepSeek v3
    """
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # DeepSeek API endpoint
    url = "https://api.deepseek.com/v1/chat/completions"
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system",
            "content": """
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
            """}, 
            {"role": "user",
             "content": f"""
             Company: {company} 
             Headline: {headline}"""    
            }
        ],
        "temperature": 0,
        "max_tokens": 50,
        "stop": ["Human:", "Assistant:", "\n\n\n"]
    }
    
    time.sleep(0.125)  # 8 requests per second rate limit
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"Error calling DeepSeek API: {e}")
        return None
    except Exception as e:
        print(f"Error processing DeepSeek API response: {e}")
        return None