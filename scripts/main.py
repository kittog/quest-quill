from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Optional
from fastapi.staticfiles import StaticFiles
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

model_path = "gpt2test"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

def generer_quete(map: str, difficulty: str, target: str) -> str:
    prompt = f"Map: {map} | Difficulty: {difficulty} | Target: {target} |"
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(
        inputs,
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=10,
        temperature=0.9,
        top_k=10,
        top_p=0.92,
        do_sample=True,
    )

    generated_sequence = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_sequence

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quest Quill</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f2f2f2;
        }

        .container {
            max-width: 480px; /* Largeur fixe pour l'apparence mobile */
            margin: 1px auto;
            background-color: #fff;
            border-radius: 10px;
            box-shadow: 0px 0px 10px 0px rgba(0,0,0,0.2);
            padding: 20px;
            margin-top: 20px;
            position: relative;
            text-align: center; /* Centrer le contenu */
        }
        .logo {
            max-width: 65%; /* Ajustement de la largeur du logo */
            
            margin: 0px auto; /* Réduction de l'espace autour du logo */
        }
        .image-container {
            text-align: center;
            margin-bottom: 20px; /* Espacement entre l'image et le formulaire */
        }
        input[type="text"],
        input[type="number"],
        input[type="submit"] {
            width: 90%; /* Définition de la largeur à 90% */
            padding: 12px; /* Augmentation de l'espacement interne */
            margin: 10px 0;
            border: none;
            border-radius: 25px; /* Bords arrondis */
            box-sizing: border-box;
            background-color: #f9f9f9; /* Couleur de fond des champs de saisie */
            font-size: 16px;
            outline: none;
        }
        input[type="submit"] {
            width: 50%; /* Largeur du bouton */
            background-color: #4CAF50;
            color: white;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        input[type="submit"]:hover {
            background-color: #45a049;
        }
        .navbar {
            display: flex;
            justify-content: space-between;
            background-color: #4CAF50;
            border-radius: 10px;
            margin-bottom: 20px;
            padding: 10px;
            overflow: hidden;
        }
        .navbar a {
            color: white;
            text-decoration: none;
            padding: 10px 15px;
            border-radius: 15px; /* Bords arrondis */
            transition: background-color 0.3s;
        }
        .navbar a:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>

<div class="container">
    
    <img src="/static/logo.png" alt="logo" class="logo">
<div class="navbar" style="display: flex; justify-content: center;">
    <a href="https://kittog.github.io/quest-quill/#generate">Manuel Quest Quill</a>
    <a href="https://kittog.github.io/quest-quill/#contact">Contact</a>
</div>

    <div class="image-container">
        <img src="/static/bann.png" alt="image" style="max-width: 100%;">
    </div>
    <form action="/generate_quest/" method="post">
        <input type="text" id="map" name="map" placeholder="Enter the map...">
        <input type="text" id="difficulty" name="difficulty" placeholder="Enter the difficulty...">
        <input type="text" id="target" name="target" placeholder="Enter the target...">
        <input type="submit" value="Generate the quest">
    </form>
</div>

</body>
</html>

    """
from fastapi.responses import HTMLResponse
@app.post("/generate_quest/", response_class=HTMLResponse)
async def generate_quest(map: str = Form(...),
                         difficulty: str = Form(...),
                         target: Optional[str] = Form(None)):
    quest_generated = generer_quete(map, difficulty, target)

    start_index = quest_generated.find("Objective")
    end_index = quest_generated.find("Map", start_index)  # Recherche à partir de start_index
    if start_index != -1 and end_index != -1:  # Vérification que les deux mots ont été trouvés
        quest_generated = quest_generated[start_index:end_index]  # Extraction de la sous-chaîne

    
    styled_quest = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Generated Quest</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f2f2f2;
                color: #ffffff;
                margin: 0;
                padding: 0;
            }}
            .container {{
                max-width: 480px; /* Largeur fixe pour l'apparence mobile */
                margin: 1px auto;
                background-color:#fff;
                border-radius: 10px;
                box-shadow: 0px 0px 10px 0px rgba(0,0,0,0.2);
                padding: 20px;
                margin-top: 20px;
                position: relative;
                text-align: center; /* Centrer le contenu */
                color: black;
            }}
            .logo {{
                max-width: 65%; /* Ajustement de la largeur du logo */
                margin: 0px auto; /* Réduction de l'espace autour du logo */
            }}
            .image-container {{
                text-align: center;
                margin-bottom: 20px; /* Espacement entre l'image et le formulaire */
            }}
            input[type="text"],
            input[type="number"],
            input[type="submit"] {{
                width: 90%; /* Définition de la largeur à 90% */
                padding: 12px; /* Augmentation de l'espacement interne */
                margin: 10px 0;
                border: none;
                border-radius: 25px; /* Bords arrondis */
                box-sizing: border-box;
                background-color: gray; /* Couleur de fond des champs de saisie */
                font-size: 16px;
                outline: none;
            }}
            input[type="submit"] {{
                width: 50%; /* Largeur du bouton */
                background-color: gray;
                color: black;
                cursor: pointer;
                transition: background-color 0.3s;
            }}
            input[type="submit"]:hover {{
                background-color: #45a049;
            }}
            .navbar {{
                display: flex;
                justify-content: space-between;
                background-color: gray;
                border-radius: 10px;
                margin-bottom: 20px;
                padding: 10px;
                overflow: hidden;
            }}
            .navbar a {{
                color: black;
                text-decoration: none;
                padding: 10px 15px;
                border-radius: 15px; /* Bords arrondis */
                transition: background-color 0.3s;
            }}
            .navbar a:hover {{
                background-color: gray;
            }}
            .quest-details {{
                margin-bottom: 20px;
                color: black;
            }}
            .quest-description {{
                padding: 20px;
                background-color: gray;
                border-radius: 10px;
                box-shadow: 0px 0px 10px 0px rgba(255, 255, 255, 0.1);
            }}
            .button-container {{
                text-align: center;
            }}
            .generate-button {{
                padding: 10px 20px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                transition: background-color 0.3s;
            }}
            .generate-button:hover {{
                background-color: #45a049;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Generated Quest</h1>
            <div class="image-container">
                <img src="/static/quest.jpg" alt="Quest" style="max-width: 100%;">
            </div>
            <div class="quest-details">
                <strong>Map:</strong> {map}<br>
                <strong>Difficulty:</strong> {difficulty}<br>
                <strong>Target:</strong> {target}
            </div>
            <div class="quest-description">
                <h2 style="text-align: center;">Quest Description</h2>
                <p>{quest_generated}</p>
            </div>

    </body>
    </html>
    '''
    
    return styled_quest
