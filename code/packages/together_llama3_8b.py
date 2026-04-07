import time
import backoff
from together import Together

@backoff.on_exception(
    backoff.expo,
    (Exception),  
    max_tries=5
)
def query_together_llama(headline, company, model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo-128K", api_key=None):
    """
    Query Llama3-8B-Instruct-Turbo-128K through Together API
    """
        
    client = Together(api_key=api_key)
    
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user", 
            "content": f"""
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
                 """
        }],
        temperature=0, 
        max_tokens=50
    )
    
    time.sleep(1)  # 1 request per second rate limit
    
    return response.choices[0].message.content