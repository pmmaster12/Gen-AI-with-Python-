from google import genai
from dotenv import load_dotenv
load_dotenv()
import os
from google.genai import types
import json
client = genai.Client()

system_prompt = """
You are an AI assistant who is expert in breaking down complex problems and then resolve the user query.

For the given user input, analyse the input and break down the problem step by step.
Atleast think 5-6 steps on how to solve the problem before solving it down.

The steps are you get a user input, you analyse, you think, you again think for several times and then return an output with explanation and then finally you validate the output as well before giving final result.

Follow the steps in sequence that is "analyse", "think", "output", "validate" and finally "result".

Rules:
1. Follow the strict JSON output as per Output schema.
2. Always perform one step at a time and wait for next input
3. Carefully analyse the user query

Output Format:
{{ step: "string", content: "string" }}

Example:
Input: What is 2 + 2.
Output: {{ step: "analyse", content: "Alright! The user is intersted in maths query and he is asking a basic arthermatic operation" }}
Output: {{ step: "think", content: "To perform the addition i must go from left to right and add all the operands" }}
Output: {{ step: "output", content: "4" }}
Output: {{ step: "validate", content: "seems like 4 is correct ans for 2 + 2" }}
Output: {{ step: "result", content: "2 + 2 = 4 and that is calculated by adding all numbers" }}

"""

query =input(">")
messages = [
    { "role": "user", "parts": [ { "text": query } ] }
]

while True:
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.5,
            response_mime_type="application/json",  # or "text/plain"
        ),
        contents=messages  # include full history each turn
    )

    model_text = resp.text  # the model's reply text (JSON string when response_mime_type is application/json)
    model_json = json.loads(model_text)
    # IMPORTANT: Store model's response with "model" role (not "user")
    # Conversation must alternate: user → model → user → model
    messages.append({ "role": "model", "parts": [ { "text": model_json.get("content", "") } ] })
    if model_json.get("step") != "result":
        print(f"🧠: {model_json.get('content')}")
        # The system prompt says "wait for next input", so we need a user message
        # to signal the model to continue to the next step
        messages.append({ "role": "user", "parts": [ { "text": "continue" } ] })
        continue
    
    print(f"🤖: {model_json.get('content')}")
    break

    # Break condition for your loop, or add next user turn:
    # next_user = input("> ")
    # messages.append({ "role": "user", "parts": [ { "text": next_user } ] })
    break