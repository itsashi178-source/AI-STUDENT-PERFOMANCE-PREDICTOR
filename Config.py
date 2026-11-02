"""Central settings. Change model names here (or with environment variables)
if Groq renames or retires a model: https://console.groq.com/docs/models
"""
import os

APP_TITLE = "AI Student Performance Predictor"

# Groq-hosted models (all three use the same GROQ_API_KEY)
LLM_MODEL = os.environ.get("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
CODING_MODEL = os.environ.get("GROQ_CODING_MODEL", "llama-3.3-70b-versatile")
VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

# Colour palette used in charts and cards
PURPLE, BLUE, PINK, TEAL, AMBER, RED = "#7c3aed", "#3b82f6", "#ec4899", "#14b8a6", "#f59e0b", "#ef4444"
