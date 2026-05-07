from dotenv import load_dotenv
import os

print("Before loading:", os.getenv("OPENAI_API_KEY"))

load_dotenv(dotenv_path=".env")

print("After loading:", os.getenv("OPENAI_API_KEY"))