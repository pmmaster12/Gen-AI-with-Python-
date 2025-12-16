from google import genai
from dotenv import load_dotenv
load_dotenv()
import os
from google.genai import types

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        system_instruction="You are an mathematical expert only answer questions related to mathematics."),
    contents="2+2"
)

print(response.text)