"""
  pip i bytez
"""
import os

# Load .env file manually
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip())

from bytez import Bytez

# Use environment variable to avoid hardcoding the API key
key = os.getenv("BYTEZ_API_KEY")
if not key:
    raise ValueError(
        "API key not found. Please set the 'BYTEZ_API_KEY' environment variable.\n"
        "Example (Windows): set BYTEZ_API_KEY=your-api-key-here\n"
        "Example (Linux/Mac): export BYTEZ_API_KEY=your-api-key-here"
    )
sdk = Bytez(key)

# choose gemini-2.5-flash
model = sdk.model("google/gemini-2.5-flash")

# send input to model
results = model.run([
  {
    "role": "user",
    "content": "Hello"
  }
])

print({ "error": results.error, "output": results.output })
