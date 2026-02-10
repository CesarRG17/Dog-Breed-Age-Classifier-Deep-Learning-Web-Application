# train_dog_classifier.py
import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

# ---------------- CONFIG ----------------
BATCH_SIZE = 32
NUM_EPOCHS = 20
IMG_SIZE = 224
IMG_DIR = ""                 # si tus image_path en CSV son relativos a una carpeta, ponla aquí; si no, deja ""
CSV_FILE = "dogs_dataset.csv" # cambia si tu csv se llama distinto (p. ej. Entrenamiento.csv)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4              # ajusta si estás en Windows/colab/servidor
LR = 1e-4
WEIGHT_AGE = 0.5             # ponderación de la pérdida de edad respecto a raza
SEED = 42
SAVE_MODEL_PATH = "dog_classifier_final.pt"
META_JSON = "dataset_meta.json"
# ----------------------------------------

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------- Dataset ----------------
class DogDataset(Dataset):
    def __init__(self, dataframe, root_dir="", transform=None, breed2id=None, age2id=None):
        """
        dataframe: pd.DataFrame con columnas ['image_path','breed','age_range' ...(opcional 'split')]
        root_dir: prefijo para rutas relativas. Si image_path es absoluto, se usa tal cual.
        transform: torchvision transforms
        breed2id, age2id: diccionarios opcionales para asegurar mapeos consistentes
        """
        self.df = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

        # Determinar mapeos si no fueron dados
        if breed2id is None:
            breeds = sorted(self.df['breed'].unique().tolist())
            self.breed2id = {b: i for i, b in enumerate(breeds)}
        else:
            self.breed2id = breed2id

        if age2id is None:
            ages = sorted(self.df['age_range'].unique().tolist())
            self.age2id = {a: i for i, a in enumerate(ages)}
        else:
            self.age2id = age2id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = str(row["image_path"])

        # Ruta absoluta o relativa
        if os.path.isabs(path) or self.root_dir == "":
            img_path = path
        else:
            img_path = os.path.join(self.root_dir, path)

        # Abrir imagen (manejo básico de errores)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # En caso de error, crear un placeholder negro (para no romper el entrenamiento)
            print(f"⚠️ No se pudo abrir {img_path}: {e}. Usando imagen placeholder.")
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0,0,0))

        if self.transform:
            image = self.transform(image)

        breed_label = self.breed2id[row["breed"]]
        age_label = self.age2id[row["age_range"]]

        return image, breed_label, age_label

# --------------- Modelo ------------------
class DogClassifier(nn.Module):
    def __init__(self, num_breeds, num_ages):
        super().__init__()
        self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # usamos la representación antes del fc
        self.breed_head = nn.Linear(in_features, num_breeds)
        self.age_head = nn.Linear(in_features, num_ages)

    def forward(self, x):
        feats = self.backbone(x)
        breed_out = self.breed_head(feats)
        age_out = self.age_head(feats)
        return breed_out, age_out

# ------------- Utilities -----------------
def prepare_dataframes(csv_file):
    df = pd.read_csv(csv_file)
    # Asegurarnos de que existan columnas mínimas
    required = ["image_path", "breed", "age_range"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"CSV requiere columna '{c}' (no encontrada).")

    # Si el csv ya contiene 'split', lo respetamos (espera 'train','val','test')
    if 'split' in df.columns:
        train_df = df[df['split'] == 'train'].reset_index(drop=True)
        val_df = df[df['split'] == 'val'].reset_index(drop=True)
        test_df = df[df['split'] == 'test'].reset_index(drop=True) if 'test' in df['split'].unique() else None
    else:
        # Estratificado por combinación raza+edad
        df['strata'] = df['breed'].astype(str) + "_" + df['age_range'].astype(str)
        train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['strata'], random_state=SEED)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['strata'], random_state=SEED)
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        df.drop(columns=['strata'], inplace=True)

    return train_df, val_df, test_df, df

def compute_weights(labels):
    # labels: array-like de ints
    classes = np.unique(labels)
    cw = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    # compute_class_weight returns in order of classes, map to full range
    weight_tensor = torch.zeros(int(classes.max())+1, dtype=torch.float)
    for cls, w in zip(classes, cw):
        weight_tensor[int(cls)] = float(w)
    return weight_tensor

# ---------------- Training ----------------
def entrenar():
    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Preparar dataframes
    train_df, val_df, test_df, full_df = prepare_dataframes(CSV_FILE)

    # Crear mapeos fijos para reproducibilidad (ordenados)
    breeds_ordered = sorted(full_df['breed'].unique().tolist())
    ages_ordered = sorted(full_df['age_range'].unique().tolist())
    breed2id = {b: i for i, b in enumerate(breeds_ordered)}
    age2id = {a: i for i, a in enumerate(ages_ordered)}

    num_breeds = len(breeds_ordered)
    num_ages = len(ages_ordered)
    print(f"➡️ Razas ({num_breeds}): {breeds_ordered}")
    print(f"➡️ Rangos edad ({num_ages}): {ages_ordered}")

    # Datasets y loaders
    train_dataset = DogDataset(train_df, root_dir=IMG_DIR, transform=train_transform, breed2id=breed2id, age2id=age2id)
    val_dataset = DogDataset(val_df, root_dir=IMG_DIR, transform=val_transform, breed2id=breed2id, age2id=age2id)
    test_dataset = DogDataset(test_df, root_dir=IMG_DIR, transform=val_transform, breed2id=breed2id, age2id=age2id) if test_df is not None else None

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS) if test_dataset is not None else None

    # Modelo
    model = DogClassifier(num_breeds=num_breeds, num_ages=num_ages).to(DEVICE)

    # Pesos por clase (opcional, útil si algo está desbalanceado)
    # Para breed:
    train_breed_labels = [breed2id[b] for b in train_df['breed'].tolist()]
    breed_weight = compute_weights(np.array(train_breed_labels))
    # Para age:
    train_age_labels = [age2id[a] for a in train_df['age_range'].tolist()]
    age_weight = compute_weights(np.array(train_age_labels))

    # Pierdas con pesos
    criterion_breed = nn.CrossEntropyLoss(weight=breed_weight.to(DEVICE))
    criterion_age = nn.CrossEntropyLoss(weight=age_weight.to(DEVICE))

    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Tracking
    train_accs_breed = []
    train_accs_age = []
    val_accs_breed = []
    val_accs_age = []

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        model.train()
        running_correct_breed = 0
        running_correct_age = 0
        running_total = 0

        for images, breed_labels, age_labels in train_loader:
            images = images.to(DEVICE)
            breed_labels = breed_labels.to(DEVICE)
            age_labels = age_labels.to(DEVICE)

            optimizer.zero_grad()
            breed_out, age_out = model(images)

            loss_breed = criterion_breed(breed_out, breed_labels)
            loss_age = criterion_age(age_out, age_labels)
            loss = loss_breed + WEIGHT_AGE * loss_age

            loss.backward()
            optimizer.step()

            # métricas
            _, preds_breed = torch.max(breed_out, 1)
            _, preds_age = torch.max(age_out, 1)
            running_correct_breed += torch.sum(preds_breed == breed_labels).item()
            running_correct_age += torch.sum(preds_age == age_labels).item()
            running_total += breed_labels.size(0)

        train_acc_breed = running_correct_breed / running_total
        train_acc_age = running_correct_age / running_total
        train_accs_breed.append(train_acc_breed)
        train_accs_age.append(train_acc_age)
        print(f" Train - Breed acc: {train_acc_breed:.4f} | Age acc: {train_acc_age:.4f}")

        # Validación
        model.eval()
        val_correct_breed = 0
        val_correct_age = 0
        val_total = 0
        with torch.no_grad():
            for images, breed_labels, age_labels in val_loader:
                images = images.to(DEVICE)
                breed_labels = breed_labels.to(DEVICE)
                age_labels = age_labels.to(DEVICE)

                breed_out, age_out = model(images)
                _, preds_breed = torch.max(breed_out, 1)
                _, preds_age = torch.max(age_out, 1)

                val_correct_breed += torch.sum(preds_breed == breed_labels).item()
                val_correct_age += torch.sum(preds_age == age_labels).item()
                val_total += breed_labels.size(0)

        val_acc_breed = val_correct_breed / val_total
        val_acc_age = val_correct_age / val_total
        val_accs_breed.append(val_acc_breed)
        val_accs_age.append(val_acc_age)
        print(f" Val   - Breed acc: {val_acc_breed:.4f} | Age acc: {val_acc_age:.4f}")

    # Guardar modelo
    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print(f"\n✅ Modelo guardado como {SAVE_MODEL_PATH}")

    # Guardar metadatos (breeds y ages)
    meta = {
        "breeds_ordered": breeds_ordered,
        "breed2id": breed2id,
        "ages_ordered": ages_ordered,
        "age2id": age2id
    }
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"✅ Metadatos guardados en {META_JSON}")

    # Evaluación sobre validation (y test si existe)
    def evaluate_and_plot(loader, split_name="val"):
        model.eval()
        y_true_b = []
        y_pred_b = []
        y_true_a = []
        y_pred_a = []
        with torch.no_grad():
            for images, breed_labels, age_labels in loader:
                images = images.to(DEVICE)
                breed_labels = breed_labels.to(DEVICE)
                age_labels = age_labels.to(DEVICE)
                breed_out, age_out = model(images)
                _, preds_b = torch.max(breed_out, 1)
                _, preds_a = torch.max(age_out, 1)

                y_true_b.extend(breed_labels.cpu().numpy())
                y_pred_b.extend(preds_b.cpu().numpy())
                y_true_a.extend(age_labels.cpu().numpy())
                y_pred_a.extend(preds_a.cpu().numpy())

        # Breed confusion
        cm_b = confusion_matrix(y_true_b, y_pred_b, labels=list(range(num_breeds)))
        disp_b = ConfusionMatrixDisplay(confusion_matrix=cm_b, display_labels=breeds_ordered)
        disp_b.plot(figsize=(10,10), xticks_rotation=90)
        plt.title(f"Matriz de confusión - Breed ({split_name})")
        plt.tight_layout()
        plt.savefig(f"cm_breed_{split_name}.png")
        plt.close()

        # Age confusion
        cm_a = confusion_matrix(y_true_a, y_pred_a, labels=list(range(num_ages)))
        disp_a = ConfusionMatrixDisplay(confusion_matrix=cm_a, display_labels=ages_ordered)
        disp_a.plot(cmap=plt.cm.Blues, xticks_rotation=45)
        plt.title(f"Matriz de confusión - Age ({split_name})")
        plt.tight_layout()
        plt.savefig(f"cm_age_{split_name}.png")
        plt.close()

        acc_b = accuracy_score(y_true_b, y_pred_b)
        acc_a = accuracy_score(y_true_a, y_pred_a)
        print(f"\n➡️ {split_name.upper()} - Breed acc: {acc_b:.4f} | Age acc: {acc_a:.4f}")
        return acc_b, acc_a

    # Evaluar en validation
    evaluate_and_plot(val_loader, split_name="val")
    # Evaluar en test si existe
    if test_loader is not None:
        evaluate_and_plot(test_loader, split_name="test")

    # Guardar curvas de accuracy
    plt.figure()
    plt.plot(range(1, NUM_EPOCHS+1), train_accs_breed, label="Train Breed")
    plt.plot(range(1, NUM_EPOCHS+1), val_accs_breed, label="Val Breed")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Breed accuracy por época")
    plt.savefig("breed_accuracy_epochs.png")
    plt.close()

    plt.figure()
    plt.plot(range(1, NUM_EPOCHS+1), train_accs_age, label="Train Age")
    plt.plot(range(1, NUM_EPOCHS+1), val_accs_age, label="Val Age")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Age accuracy por época")
    plt.savefig("age_accuracy_epochs.png")
    plt.close()

if __name__ == "__main__":
    entrenar()
