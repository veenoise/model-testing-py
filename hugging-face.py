from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import time
import base64

load_dotenv()
HUGGING_FACE_API_KEY=os.getenv('HUGGING_FACE_API_KEY')

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

base64_image = encode_image_to_base64('./aircon.jpg')
# data_url = f"data:image/jpeg;base64,{base64_image}"

data_url = 'https://7227-103-91-141-171.ngrok-free.app/aircon.jpg'
JSON_INSTRUCTIONS = '''
You are a strict real-time activity validator.

You will receive a short, time-ordered sequence of frames and a list of tasks (with stable IDs).
For EACH task, return a JSON object keyed by the task's "id", with fields:

{
  "<task_id>": {
    "active": boolean,        // true if action is ongoing in these frames
    "completed": boolean,     // true if the task result appears finished/visible
    "confidence": number,     // 0..1
    "notes": string           // short rationale or why it's uncertain
  },
  ...
}

Rules:
- ONLY include task IDs from "allowed_ids".
- If uncertain, set active=false, completed=false, confidence <= 0.35, and add a short "notes".
- Prefer "completed" only when an end-state is visible (e.g., clearly cleaned blades).
- Never invent task IDs; never include extra fields.
- All provided frames should be interpreted as first-person view from a technician's body-worn camera. Treat camera angles and visible hands/tools as belonging to the technician. Do not assume third-person perspectives. Only evaluate actions visible from this POV.
'''

client = InferenceClient(api_key=HUGGING_FACE_API_KEY)
model = "baidu/ERNIE-4.5-VL-28B-A3B-PT:novita"

start_time = time.time()

completion = client.chat.completions.create(
    model=model, 
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""
{JSON_INSTRUCTIONS}
\n\nallowed_ids:
[
    "clean_indoor_unit_filters",
    "inspect_evaporator_coil",
    "check_air_flow",
    "inspect_refrigerant_lines",
    "clean_outdoor_condenser_fins",
    "tighten_electrical_terminals",
    "check_thermostat_calibration"
]
\n\ntasks:\n
[
    {{
    "id": "clean_indoor_unit_filters",
    "label": "Clean indoor unit filters",
    "description": "A first-person view of the AC indoor unit with the filter visible.",
    }},
    {{
    "id": "inspect_evaporator_coil",
    "label": "Inspect evaporator coil",
    "description": "A first-person view showing the AC unit's evaporator coil.",
    }},
    {{
    "id": "check_air_flow",
    "label": "Check air flow",
    "description": "First-person view showing the user's hand in front of the split-type aircon.",
    }},
    {{
    "id": "inspect_refrigerant_lines",
    "label": "Inspect refrigerant lines",
    "description": "A first-person view of the refrigerant pipes near the indoor or outdoor unit.",
    }},
    {{
    "id": "clean_outdoor_condenser_fins",
    "label": "Clean outdoor condenser fins",
    "description": "A first-person view of the outdoor condenser fins.",
    }},
    {{
    "id": "tighten_electrical_terminals",
    "label": "Tighten electrical terminals",
    "description": "A first-person view of the opened electrical terminal panel of the AC unit.",
    }},
    {{
    "id": "check_thermostat_calibration",
    "label": "Check thermostat calibration",
    "description": "A first-person view of the thermostat mounted on the wall.",
    }}
]
"""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    }
                }
            ]
        }
    ]
)

end_time = time.time()
response_time = end_time - start_time

try:
    print(f"{completion.choices[0].message.content}")
    print(f"INPUT: {completion.usage.prompt_tokens}")
    print(f"OUTPUT: {completion.usage.completion_tokens}")
    print(f"Response time: {response_time:.3f} seconds")

    with open(f"./logs/{model.replace("/", "-")}.log", "w") as file:
        file.write(f"{completion.choices[0].message.content}\n")
        file.write(f"INPUT: {completion.usage.prompt_tokens}\n")
        file.write(f"OUTPUT: {completion.usage.completion_tokens}\n")
        file.write(f"Response time: {response_time:.3f} seconds")
except:
  print(completion)

