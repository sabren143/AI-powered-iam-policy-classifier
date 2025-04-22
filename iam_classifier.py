from openai import OpenAI
from transformers import pipeline # for Hugging face LLM

import json

client = OpenAI(api_key="Paste your openai API key here")

def classify_policy_openai(input_filename, output_filename="result.json"):
    with open(input_filename, "r") as f:
        policy_json = json.load(f)
        prompt = f"""
    You are a cloud security expert.

    Classify the following IAM policy as either "Weak" or "Strong" and explain why.

    Respond in this JSON format:
    {{
      "policy": <copy of the input policy>,
      "classification": "...",
      "reason": "..."
    }}

    IAM Policy:
    {json.dumps(policy_json, indent=2)}
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    result_text = response.choices[0].message.content

    try:
        result_json = json.loads(result_text)
    except json.JSONDecodeError:
        result_json = {
            "error": "Model returned invalid JSON.",
            "raw_output": result_text
        }

    with open(output_filename, "w") as f:
        json.dump(result_json, f, indent=2)

    return output_filename



def classify_policy_huggingface(input_filename, output_filename="result_flan.json"):
    with open(input_filename, "r") as f:
        policy_json = json.load(f)

    classifier = pipeline("text2text-generation", model="google/flan-t5-xl")

    prompt = f"""
You are a cloud security expert.

Classify the following IAM policy as either "Weak" or "Strong" and explain why.

Respond in this JSON format:
{{
  "policy": <copy of the input policy>,
  "classification": "...",
  "reason": "..."
}}

IAM Policy:
{json.dumps(policy_json, indent=2)}
"""

    result_text = classifier(prompt, max_new_tokens=300)[0]['generated_text']

    try:
        result_json = json.loads(result_text)
    except json.JSONDecodeError:
        result_json = {
            "error": "FLAN returned invalid JSON.",
            "raw_output": result_text
        }

    with open(output_filename, "w") as f:
        json.dump(result_json, f, indent=2)

    return output_filename

"""
if __name__ == "__main__":

    # Run OpenAI
    output_openai = classify_policy_openai("policy.json", "result_openai.json")
    print(f"[OpenAI] Output saved to: {output_openai}")

    # Run Hugging Face
    output_hf = classify_policy_huggingface("policy.json", "result_huggingface.json")
    print(f"[Hugging Face] Output saved to: {output_hf}")

"""