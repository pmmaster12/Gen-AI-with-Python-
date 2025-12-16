# structure
# '''
#  user ask query in nlp about weather of city so you should determine city from it and then decide which 
#  tool (function calling) should use and then call the function get the response and return to user
# '''
from dotenv import load_dotenv
import os
import json 
import requests
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client()

def get_weather(city : str):
    print(f"🔨:  Tool called get_weather for city {city}")
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)
    if response.status_code == 200:
        return f"The weather in {city} is {response.text}."
    return "Something went wrong"
        

def run_command(command: str):
    result = os.system(command=command)
    return result
        
        

avaiable_tools = {
    "get_weather": {
        "fn": get_weather,
        "description": "Takes a city name as an input and returns the current weather for the city"
    },
    "run_command": {
        "fn": run_command,
        "description": "Takes a command as input to execute on system and returns ouput"
    }
}

system_prompt = f"""
    You are an helpfull AI Assistant who is specialized in resolving user query.    
    You work on analyze, plan, action, observe mode.
    For the given user query and available tools, analyse the user query and plan and reason upon which of the available tool should be appropriate
    For the given user query and available tools, plan the step by step execution, based on the analysis,
    select the relevant tool from the available tool. and based on the tool selection you perform an action to call the tool.
    Wait for the observation and based on the observation from the tool call resolve the user query.
    If you don't find any of the available tools useful for resolving query then you should use your own reasoning for answering the query
    
    Rules:
    - Follow the Output JSON Format.
    - Always perform one step at a time and wait for next input
    - Carefully analyse the user query
    - To execute command always come up with python or windows equivalent command don't generate linux commands

    Output JSON Format:
    {{
        "step": "string",
        "content": "string",
        "function": "The name of function if the step is action",
        "input": "The input parameter for the function",
    }}

    Available Tools:
    - get_weather: Takes a city name as an input and returns the current weather for the city
    - run_command: Takes a command as input to execute on system and returns ouput
    
    Example:
    User Query: What is the weather of new york?
    Output: {{ "step": "analyse", "content": "The user wanted to know about the real time weather information of new york city to get this it would need real time weather data so search upon the available tools and find any tool relevant for this query" }}
    Output: {{ "step": "plan", "content": "The user is interseted in weather data of new york and From the available tools I should call get_weather" }}
    Output: {{ "step": "action", "function": "get_weather", "input": "new york" }}
    Output: {{ "step": "observe", "output": "12 Degree Cel" }}
    Output: {{ "step": "output", "content": "The weather for new york is 12 degrees." }}
"""

query =input(">")
messages = [
    { "role": "user", "parts": [ { "text": query } ] }
]

while True:
    resp = client.models.generate_content(
        model="gemini-2.5-pro",
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
        
    if model_json.get("step",None) == "analyse":
        print(f"🧠: {model_json.get('step')} - {model_json.get('content')}")
        messages.append({ "role": "user", "parts": [ { "text": "continue" } ] })
        print("1")
        continue
        
    
    if model_json.get("step",None) == "plan":
        print(f"🧠: {model_json.get('step')} - {model_json.get('content')}")
        messages.append({ "role": "user", "parts": [ { "text": "continue" } ] })
        print("2")
        continue
    
     
    if model_json.get("step",None) == "action":
        print(f"🧠: {model_json.get('step')} - {model_json.get('function')} - {model_json.get('input')}")
        tool_name = model_json.get("function")
        tool_parameter = model_json.get("input")
        
        if avaiable_tools.get(tool_name):
            output = avaiable_tools.get(tool_name).get("fn")(tool_parameter)
            messages.append({ "role": "model", "parts": [ { "text": json.dumps({ "step": "observe", "output":  output}) } ] })
            messages.append({ "role": "user", "parts": [ { "text": "continue" } ] })
            print("3")
            continue
        
    if model_json.get("step") == "output":
        print(f"🤖: {model_json.get('step')} - {model_json.get('content')}")
        break