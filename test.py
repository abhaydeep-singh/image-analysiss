import ollama
import base64
from pathlib import Path

def analyze_image(image_path: str, question: str) -> str:
    # Read and encode image to base64
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    
    response = ollama.chat(
        model="moondream",
        messages=[
            {
                "role": "user",
                "content": question,
                "images": [image_data]
            }
        ]
    )
    
    return response["message"]["content"]


image_path = "test.jpg"

result = analyze_image(image_path, "Describe what you see in this image.")
print("Basic description:\n", result)

prompt = """
Analyze this image from a forensic investigation perspective.
Identify:
1. People present and what they are doing
2. All visible objects
3. Location/environment
4. Any suspicious activity or items
5. Suspicion score from 1-10 with reason

Be factual and specific. Use words like 'possible' and 'suspected' 
rather than making definitive conclusions.
"""

result = analyze_image(image_path, prompt)
print("\Analysis: \n", result)