import io
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from torchvision.models import efficientnet_b3

app = FastAPI()

# Configuración de CORS para evitar bloqueos del navegador
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 1. ARQUITECTURA DEL MODELO ====================

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
        base = efficientnet_b3(weights=None)
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

# ==================== 2. CARGA DE MODELO Y RECURSOS ====================

DEVICE = torch.device("cpu") # Forzamos CPU para Render (evita errores de memoria)
MODEL_PATH = "final_multitask_v2.pt"
CSV_PATH = "Pesos_Final.csv"
NO_DOG_THRESHOLD = 0.25

try:
    df_weights = pd.read_csv(CSV_PATH)
    breeds_list = sorted(df_weights['Breed'].unique().tolist())
    age_ranges = ["0-1", "1-4", "4-8", "8-13"]

    model = MultiTaskEfficientNet(num_breeds=len(breeds_list), num_age_classes=len(age_ranges))
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print("✅ Recursos cargados exitosamente.")
except Exception as e:
    print(f"❌ Error al cargar recursos: {e}")

img_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==================== 3. RUTAS Y ENDPOINTS ====================

# Sirve la página principal al entrar a la URL
@app.get("/")
async def read_index():
    return FileResponse('index.html')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_tensor = img_transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out_breed, out_age = model(img_tensor)
            probs_b = torch.softmax(out_breed, dim=1)
            confidence, idx_b = torch.max(probs_b, 1)
            idx_a = torch.argmax(out_age, 1)

        conf_val = confidence.item()
        breed_key = breeds_list[idx_b.item()]
        age_key = age_ranges[idx_a.item()]

        if conf_val < NO_DOG_THRESHOLD:
            return {
                "is_dog": False,
                "message": "⚠️ La confianza es baja. No parece una raza conocida.",
                "confidence": f"{conf_val:.2%}"
            }

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
        return {"error": str(e)}

# ==================== 4. CONFIGURACIÓN DE PUERTO ====================

if __name__ == "__main__":
    import uvicorn
    # Render inyecta el puerto en la variable de entorno PORT
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
