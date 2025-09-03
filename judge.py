import anthropic
import openai
import os 
import pandas as pd
import numpy as np
import re
import json
import requests
import time

model_name_full_to_short = {
    "claude-sonnet-4-20250514": "claude",
    "gpt-4.1-mini-2025-04-14": "gpt-4-1-mini"
}

model_name_short_to_full = {
    "claude": "claude-sonnet-4-20250514",
    "gpt-4-1-mini": "gpt-4.1-mini-2025-04-14"
}

TOPIC_CLASSIFIER_SYSTEM_PROMPT = """
# Instruction
Your job is to classify the following **user query** into one of the following task categories. Base your judgment on both the user's intent and the AI assistant's response. The query may be part of a conversation, so you may have to infer the task category based on the AI assistant's response. 

**Task Categories:**
- A: Factual Information - The user is asking for knoweldge, facts, or explanations (like using a search engine).
- B: Professional or Work-Related Writing: The user wants help drafting or improving work/professional content (emails, reports, resumes, presentations, etc.).
- C: Coding Assistance or Technical Tasks: The user is asking for programming help, debugging, technical setup, or data analysis. 
- D: Personal or Social Advice: The user is seeking guidance about life, relationships, health, or personal situations.
- E: Recommendations for Hobbies or Leisure: The user wants suggestions for books, movies, games, travel, activities, or entertainment. 
- F: Understanding News or Political Topics: The user is asking about current events, policies, elections, or government-related topics.
- G: Other: Classify as other only as a last resort if none of the above task categories apply.  

# Input Format: You will receive:
- User's Query
- AI Assistant's Response

# Output Format
Output only the single task category letter that you assign to the user's query (e.g. A, B, C, D, E, F, or G). Do not output any other text or explanations.
""".strip()

TOPIC_CLASSIFIER_PROMPT = """User's Query: {query}\n\nAI Assistant's Response: {response}"""



class JudgeClaude:
    def __init__(self):
        self.client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )
    
    def build_batch_query(self, query_id, system_prompt, prompt):
        return {
            "custom_id": query_id,
            "params": {
                "model": "claude-sonnet-4-20250514",
                "max_tokens":  5,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
                "system": system_prompt
            }
        }
    
    def run_batch(self, jsonl_array):
        message_batch = self.client.beta.messages.batches.create(
            requests=jsonl_array
        )
        return message_batch

    def process_batch_results(self, jsonl_file="claude_results.jsonl"):
        df = []
        with open(jsonl_file, 'r') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    df.append([data["custom_id"], data["result"]["message"]["content"][0]["text"]])
                except json.JSONDecodeError as e:
                    continue
        df = pd.DataFrame(df, columns=["id", "claude_judge"])
        return df
    
    def download_batch(self, batch_id):
        batch = self.client.beta.messages.batches.retrieve(batch_id)
        headers = {
            'x-api-key': self.client.api_key,
            'anthropic-version': '2023-06-01',
            'anthropic-beta': 'message-batches-2024-09-24'
        }
        response = requests.get(batch.results_url, headers=headers)
        content = response.text
        with open("claude_results.jsonl", 'w') as f:
            f.write(content)

    def get_batch_results(self, batch_id):
        self.download_batch(batch_id)
        return self.process_batch_results()

    def run_topic_classifier(self, ids, queries, responses):
        batch = []
        for id,q,r in zip(ids, queries, responses):
            query = self.build_batch_query(id, TOPIC_CLASSIFIER_SYSTEM_PROMPT, TOPIC_CLASSIFIER_PROMPT.format(query=q, response=r))
            batch.append(query)
        batch = self.run_batch(batch)
        return batch.id

class JudgeGPT:
    def __init__(self):
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.client = openai.OpenAI()
    
    def build_batch_query(self, query_id, system_prompt, prompt):
        query = {
            "custom_id": query_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4.1-mini-2025-04-14",
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                "max_completion_tokens": 5,
                "temperature": 0
            }
        }

        return json.dumps(query)

    def run_batch(self, jsonl_file):
        batch_input_file = self.client.files.create(
            file=open(jsonl_file, "rb"),
            purpose="batch"
        )

        batch_input_file_id = batch_input_file.id

        batch_object = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        return batch_object.id

    def download_batch(self, batch_id):
        batch = self.client.batches.retrieve(batch_id)
        output_file_id = batch.output_file_id
        response = self.client.files.with_raw_response.retrieve_content(output_file_id)

        with open("gpt_results.jsonl", "w", encoding="utf-8") as f:
            f.write(response.content.decode("utf-8"))

    def process_batch_results(self, jsonl_file="gpt_results.jsonl"):
        df = []
        with open(jsonl_file, 'r') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    df.append([data["custom_id"], data["response"]["body"]["choices"][0]["message"]["content"]])
                except json.JSONDecodeError as e:
                    continue
        df = pd.DataFrame(df, columns=["id", "gpt_judge"])
        return df

    def get_batch_results(self, batch_id):
        self.download_batch(batch_id)
        return self.process_batch_results()

    def run_topic_classifier(self, ids, queries, responses):
        batch_file = "batch.jsonl"
        with open(batch_file, 'w') as file:
            for id,q,r in zip(ids, queries, responses):
                query = self.build_batch_query(id, TOPIC_CLASSIFIER_SYSTEM_PROMPT, TOPIC_CLASSIFIER_PROMPT.format(query=q, response=r))
                file.write(query + "\n")
        batch_id = self.run_batch(batch_file)
        return batch_id