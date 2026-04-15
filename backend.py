import io
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
from fastapi.middleware.cors import CORSMiddleware
from torchvision.models import efficientnet_b3

app = FastAPI()

# Permitir comunicación con el Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 1. DEFINICIÓN DE ARQUITECTURA ====================
# (Debe ser idéntica a la de tu código de entrenamiento)

class GeMPooling(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor):
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p).flatten(1)

class MultiTaskEfficientNet(nn.Module):
    def __init__(self, num_breeds: int, num_age_classes: int = 4):
        super().__init__()
        base = efficientnet_b3(weights=None) # No necesitamos pesos de imagenet aquí, cargaremos los tuyos
        self.backbone = base.features
        self.pool = GeMPooling(p=3.0)
        in_feats = 1536

        self.breed_head = nn.Sequential(
            nn.Linear(in_feats, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_breeds)
        )

        self.age_head = nn.Sequential(
            nn.Linear(in_feats, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.45),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_age_classes)
        )

    def forward(self, x: torch.Tensor):
        feats = self.pool(self.backbone(x))
        return self.breed_head(feats), self.age_head(feats)

# ==================== 2. CARGA DE MODELO Y DATOS ====================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "final_multitask_v2.pt"
CSV_PATH = "Pesos_Final.csv"
NO_DOG_THRESHOLD = 0.25 # Bajamos un poco para evitar falsos negativos en el inicio

# Instanciamos y cargamos el modelo
try:
    df_weights = pd.read_csv(CSV_PATH)
    # Extraemos razas y edades en el orden correcto
    breeds_list = sorted(df_weights['Breed'].unique().tolist())
    age_ranges = ["0-1", "1-4", "4-8", "8-13"]

    model = MultiTaskEfficientNet(num_breeds=len(breeds_list), num_age_classes=len(age_ranges))
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print("✅ Sistema listo: Modelo y CSV cargados correctamente.")
except Exception as e:
    print(f"❌ ERROR CRÍTICO: {e}")

# Transformaciones de imagen (idénticas a la validación del entrenamiento)
img_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==================== 3. ENDPOINTS ====================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Procesar imagen
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_tensor = img_transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out_breed, out_age = model(img_tensor)
            
            # Obtener probabilidades y confianza
            probs_b = torch.softmax(out_breed, dim=1)
            confidence, idx_b = torch.max(probs_b, 1)
            
            # Obtener edad
            idx_a = torch.argmax(out_age, 1)

        conf_val = confidence.item()
        breed_key = breeds_list[idx_b.item()]
        age_key = age_ranges[idx_a.item()]

        print(f"DEBUG: Predicción: {breed_key} | Confianza: {conf_val:.4f}")

        # Validación de Umbral
        if conf_val < NO_DOG_THRESHOLD:
            return {
                "is_dog": False,
                "message": "⚠️ No se reconoce como una raza conocida.",
                "confidence": f"{conf_val:.2%}"
            }

        # Buscar en CSV
        match = df_weights[(df_weights['Breed'] == breed_key) & 
                           (df_weights['Age_Group'] == age_key)]
        
        peso_final = f"{match.iloc[0]['Avg_Weight_kg']} kg" if not match.empty else "No disponible"

        return {
            "is_dog": True,
            "breed": breed_key.replace("_", " "),
            "age": f"{age_key} años",
            "weight": peso_final,
            "confidence": f"{conf_val:.2%}"
        }

    except Exception as e:
        print(f"Error en predicción: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)