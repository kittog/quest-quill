from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Optional
from fastapi.staticfiles import StaticFiles
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

model_path = "gpt2_quest_generator_finetuned2"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

def generer_quete(map: str, difficulty: str, target: str) -> str:
    prompt = f"Map: {map} | Difficulty: {difficulty} | Target: {target} ->"
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=300, num_return_sequences=1, no_repeat_ngram_size=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Quest Quill</title>
    <style>
        body {
            max-width: 800px;
            max-height: 800px
            font-family: Arial, sans-serif;
            background-color: #d3d3d3;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: auto;
            flex-direction: row;
            position: center;
            border-radius: 5px;
            box-shadow: 0px 0px 10px 0px rgba(0,0,0,0.2);
        }
        .image-container {
            margin-right: 50px;
        }
        .form-container {
            max-width: 400px;
            background: #fff;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0px 0px 10px 0px rgba(0,0,0,0.2);
        }
        input[type="text"],
        input[type="number"] {
            width: 100%;
            padding: 10px;
            margin: 5px 0;
            border: 1px solid #ccc;
            border-radius: 4px;
            box-sizing: border-box;
        }
        input[type="submit"] {
            background-color: #4CAF50;
            color: white;
            padding: 14px 20px;
            margin: 8px 0;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            width: 100%;
        }
        input[type="submit"]:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>

<div class="logo-container" style="position: absolute; top: 5px; width: 100%; text-align: center; padding: 20px 0;">
    <img src="/static/logo.png" alt="Logo" style="max-width: 500px;">
</div>
<div class="image-container">
    <img src="/static/bann.png" alt="Banner Image" style="max-width: 300px; max-height: 100vh;">
</div>
<div class="form-container">
    <h1>Quest Generator</h1>
    <form action="/generate_quest/" method="post">
        <label for="map">Map:</label>
        <input type="text" id="map" name="map" placeholder="Enter the map..."><br>
        <label for="difficulty">Difficulty:</label>
        <input type="text" id="difficulty" name="difficulty" placeholder="Enter the difficulty..."><br>
        <label for="target">Target:</label>
        <input type="text" id="target" name="target" placeholder="Enter the target..."><br>
        <input type="submit" value="Generate the quest">
    </form>
</div>

</body>
</html>

    """

@app.post("/generate_quest/")
async def generate_quest(map: str = Form(...),
                         difficulty: str = Form(...),
                         target: Optional[str] = Form(None)):
    quest_generated = generer_quete(map, difficulty, target)
    
    styled_quest = f'<div style="background-color: #ffffff; padding: 20px; border-radius: 5px; box-shadow: 0px 0px 10px 0px rgba(0,0,0,0.1); margin-top: 20px;">{quest_generated}</div>'
    
    return HTMLResponse(content=styled_quest)
