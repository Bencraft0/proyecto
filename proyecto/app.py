import os
import requests
from flask import Flask, request, render_template_string
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

app = Flask(__name__)

# 🔐 Leer API Key desde variable de entorno (no hardcodear)
IMGBB_API_KEY = os.environ.get("IMGBB_API_KEY")

labels_map = {
    "plástico": ["botella de plástico", "envase plástico", "bolsa de plástico", "film plástico", "envase PET", "envase HDPE", "bolsa plástica"],
    "papel": ["hoja de papel", "periódico", "revista", "folleto", "papel de oficina", "papel de diario", "papel reciclado"],
    "cartón": ["caja de cartón", "empaque de cartón", "cartón corrugado", "cartón de embalaje", "caja para envío"],
    "tetra pak": ["envase de tetra pak", "caja de jugo tetra pak", "envase de leche tetra pak", "cartón combinado"],
    "aluminio": ["lata de aluminio", "envase metálico", "papel aluminio", "envase de bebidas", "hoja de aluminio"],
    "material peligroso": ["pila o batería usada", "producto químico peligroso", "residuo tóxico", "aceite usado", "producto inflamable", "residuo sanitario"],
    "vidrio": ["botella de vidrio", "vaso de vidrio", "frasco de vidrio", "envase de vidrio", "vidrio transparente", "vidrio coloreado"],
    "orgánico": ["restos de comida", "residuo orgánico", "cáscara de fruta", "hojas secas", "restos vegetales", "residuo biodegradable"],
    "otro": ["otro tipo de residuo", "material mixto", "residuo no clasificado", "plásticos duros", "residuo textil"]
}

all_labels = []
label_to_primary = {}
for primary, secondary_list in labels_map.items():
    for s in secondary_list:
        all_labels.append(s)
        label_to_primary[s] = primary

HTML_FORM = """..."""  # (Todo el HTML ya lo tenés bien, así que no lo repito acá por brevedad)

def upload_to_imgbb(image_file):
    url = "https://api.imgbb.com/1/upload"
    payload = {
        "key": IMGBB_API_KEY,
    }
    files = {
        "image": image_file.read(),
    }
    response = requests.post(url, data=payload, files=files)
    if response.status_code == 200:
        return response.json()['data']['url']
    else:
        return None

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_url = None
    if request.method == "POST":
        if "image" not in request.files or request.files["image"].filename == "":
            return render_template_string(HTML_FORM, result=None, image_url=None)

        image_file = request.files["image"]

        image_url = upload_to_imgbb(image_file)
        if not image_url:
            return "Error al subir la imagen a imgbb"

        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

        inputs = processor(text=all_labels, images=image, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(**inputs)

        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)[0].tolist()

        scores_primary = {}
        for label, score in zip(all_labels, probs):
            primary = label_to_primary[label]
            scores_primary[primary] = scores_primary.get(primary, 0) + score

        result = sorted(scores_primary.items(), key=lambda x: x[1], reverse=True)

    return render_template_string(HTML_FORM, result=result, image_url=image_url)

# ✅ Configuración compatible con Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
