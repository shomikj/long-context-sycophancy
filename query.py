import anthropic
import openai
import os 
import pandas as pd
import numpy as np
import re
import json
import requests
import time

def build_context(df):
    context = []
    for _,r in df.iterrows():
        context.append({
            "role": "user",
            "content": r["input"]
        })
        context.append({
            "role": "assistant",
            "content": r["output"]
        })
    return context

class QueryClaude:
    def __init__(self):
        self.client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )

    def build_batch_query(self, query_id, context, prompt, max_tokens=200):
        message_data = context + [{
                "role": "user",
                "content": prompt,
            }]
        query = {
            "custom_id": query_id,
            "params": {
                "model": "claude-sonnet-4-20250514",
                "max_tokens":  max_tokens,
                "temperature": 0.5,
                "messages": message_data
            }
        }
        return query

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
        df = pd.DataFrame(df, columns=["id", "response"])
        df["prompt_id"] = df['id'].apply(lambda x: str(x).split('_')[0])
        df["iteration"] = df['id'].apply(lambda x: str(x).split('_')[1])
        df["context"] = df['id'].apply(lambda x: str(x).split('_')[2])
        df["model"] = "claude-sonnet-4-20250514"
        df = df.drop(columns=["id"])
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

    def build_and_run_batch(self, prompts, context_ids, contexts, num_iterations):
        batch = []
        for _,r in prompts.iterrows():
            prompt_id = r["prompt_id"]
            prompt = r["prompt"]

            for context_id, context in zip(context_ids, contexts):
                for iteration in range(num_iterations):
                    query_id = prompt_id + "_" + str(iteration) + "_" + context_id
                    query = self.build_batch_query(query_id, context, prompt)
                    batch.append(query)

        batch = self.run_batch(batch)
        return batch.id
        '''
        batch_id = batch.id
        iterations = 0
        while batch.processing_status != "ended":
            iterations += 1
            print("Waiting for batch to finish", iterations)
            time.sleep(60)
            if iterations > 10:
                return None
            batch = self.client.beta.messages.batches.retrieve(batch_id)
        
        self.download_batch(batch_id)
        return self.process_batch_results()
        '''

class QueryGPT:
    def __init__(self):
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.client = openai.OpenAI()

    def build_batch_query(self, query_id, context, prompt, max_tokens=200):
        message_data = context + [{
                "role": "user",
                "content": prompt,
            }]
        
        query = {
            "custom_id": query_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4.1-mini-2025-04-14",
                "messages": message_data,
                "max_completion_tokens": max_tokens,
                "temperature": 1.0
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
        df = pd.DataFrame(df, columns=["id", "response"])
        df["prompt_id"] = df['id'].apply(lambda x: str(x).split('_')[0])
        df["iteration"] = df['id'].apply(lambda x: str(x).split('_')[1])
        df["context"] = df['id'].apply(lambda x: str(x).split('_')[2])
        df["model"] = "gpt-4.1-mini-2025-04-14"
        df = df.drop(columns=["id"])
        return df

    def get_batch_results(self, batch_id):
        self.download_batch(batch_id)
        return self.process_batch_results()

    def build_and_run_batch(self, prompts, context_ids, contexts, num_iterations):
        batch_file = "batch.jsonl"
        with open(batch_file, 'w') as file:
            for _,r in prompts.iterrows():
                prompt_id = r["prompt_id"]
                prompt = r["prompt"]
                for context_id, context in zip(context_ids, contexts):
                    for iteration in range(num_iterations):
                        query_id = prompt_id + "_" + str(iteration) + "_" + context_id
                        query = self.build_batch_query(query_id, context, prompt)
                        file.write(query + "\n")
        batch_id = self.run_batch(batch_file)
        return batch_id
        '''
        batch = self.client.batches.retrieve(batch_id)
        iterations = 0
        while batch.status != "completed":
            iterations += 1
            print("Waiting for batch to finish", iterations)
            time.sleep(60)
            if iterations > 10:
                return None
            batch = self.client.batches.retrieve(batch_id)
        
        self.download_batch(batch_id)
        return self.process_batch_results()
        '''
