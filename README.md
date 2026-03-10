# Model Testing - AI Vision Task Validator

Test and compare vision AI models for real-time activity validation from first-person camera footage.

## Overview

This project tests multiple AI models (Google Gemini, OpenAI GPT, and OpenRouter models) to validate AC maintenance tasks from images. Each model analyzes an image and returns structured JSON indicating task completion status.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
```bash
cp sample.env .env
```

Edit `.env` with your credentials:
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to Vertex AI credentials JSON
- `OPENAI_API_KEY`: Your OpenAI API key
- `OPEN_ROUTER_API_KEY`: Your OpenRouter API key

## Usage

Test individual providers:

```bash
# Google Gemini models
python gemini.py

# OpenAI models
python openai-test.py

# OpenRouter models
python open-router.py
```

Results are saved to `logs/` with token counts and response times.

## Task Validation

Models evaluate 7 AC maintenance tasks:
- Clean indoor unit filters
- Inspect evaporator coil
- Check air flow
- Inspect refrigerant lines
- Clean outdoor condenser fins
- Tighten electrical terminals
- Check thermostat calibration

Each task returns:
- `active`: Action currently in progress
- `completed`: Task finished/visible
- `confidence`: 0-1 score
- `notes`: Rationale

## Models Tested

- **Google Gemini**: gemini-3.1-flash-lite-preview, gemini-2.5-flash, etc.
- **OpenAI**: gpt-4o, gpt-5, gpt-4.1-mini, etc.
- **OpenRouter**: Google Gemma, and Nvidia Nemotron
