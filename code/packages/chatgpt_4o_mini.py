import time
import backoff
from openai import OpenAI

@backoff.on_exception(
    backoff.expo,
    (Exception),  
    max_tries=5
)
def query_chatgpt_4o_mini(headline, company, api_key=None, model="gpt-4o-mini"):
    """
    Query GPT-4o-mini
    """

    client = OpenAI(api_key=api_key)
    
    request = {
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
        response = client.chat.completions.create(**request)
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None