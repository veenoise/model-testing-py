from google import genai
from google.genai import types
from dotenv import load_dotenv
import time

load_dotenv()

GOOGLE_CLOUD_PROJECT = "ai-innov-474401"
GOOGLE_CLOUD_LOCATION = "global"

client = genai.Client(
    vertexai=True,
    project=GOOGLE_CLOUD_PROJECT,
    location=GOOGLE_CLOUD_LOCATION,
)

with open("./aircon.jpg", "rb") as f:
    image_bytes = f.read()

image = types.Part.from_bytes(
    data=image_bytes,
    mime_type="image/jpeg",
)

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

start_time = time.time()

model = "gemini-3.1-flash-lite-preview"

response = client.models.generate_content(
    model=model,
    contents=[image, JSON_INSTRUCTIONS +
              "\n\nallowed_ids: " +
              str([
                "clean_indoor_unit_filters",
                "inspect_evaporator_coil",
                "check_air_flow",
                "inspect_refrigerant_lines",
                "clean_outdoor_condenser_fins",
                "tighten_electrical_terminals",
                "check_thermostat_calibration"
              ]) +
              "\n\ntasks:\n" +
              str([
                {
                  "id": "clean_indoor_unit_filters",
                  "label": "Clean indoor unit filters",
                  "description": "A first-person view of the AC indoor unit with the filter visible.",
                },
                {
                  "id": "inspect_evaporator_coil",
                  "label": "Inspect evaporator coil",
                  "description": "A first-person view showing the AC unit's evaporator coil.",
                },
                {
                  "id": "check_air_flow",
                  "label": "Check air flow",
                  "description": "First-person view showing the user's hand in front of the split-type aircon.",
                },
                {
                  "id": "inspect_refrigerant_lines",
                  "label": "Inspect refrigerant lines",
                  "description": "A first-person view of the refrigerant pipes near the indoor or outdoor unit.",
                },
                {
                  "id": "clean_outdoor_condenser_fins",
                  "label": "Clean outdoor condenser fins",
                  "description": "A first-person view of the outdoor condenser fins.",
                },
                {
                  "id": "tighten_electrical_terminals",
                  "label": "Tighten electrical terminals",
                  "description": "A first-person view of the opened electrical terminal panel of the AC unit.",
                },
                {
                  "id": "check_thermostat_calibration",
                  "label": "Check thermostat calibration",
                  "description": "A first-person view of the thermostat mounted on the wall.",
                }
              ])],
    config=types.GenerateContentConfig(
        tools=[types.Tool(code_execution=types.ToolCodeExecution)]
    ),
)

end_time = time.time()
response_time = end_time - start_time

print(response.text)
print("INPUT:", response.usage_metadata.prompt_token_count)
print("OUTPUT:", response.usage_metadata.candidates_token_count)
print(f"Response time: {response_time:.3f} seconds")

with open(f"./logs/{model}.log", 'w') as file:
  file.write(f"{response.text}\n")
  file.write(f"INPUT: {response.usage_metadata.prompt_token_count}\n")
  file.write(f"OUTPUT: {response.usage_metadata.candidates_token_count}\n")
  file.write(f"Response time: {response_time:.3f} seconds")