import csv


OUTPUT_CSV = "dogs_dataset.csv"
BASE_PATH = "IMAGENES"
IMAGES_PER_AGE = 50

BREEDS = [
    "Akita Inu", "Beagle", "Border Collie", "Boxer",
    "Bulldog Francés", "Bulldog Inglés", "Calupoh",
    "Caniche (Poodle)", "Chihuahua", "Cocker Spaniel",
    "Dachshund", "Dóberman", "Golden Retriever",
    "Gran Danés", "Husky Siberiano", "Labrador Retriever",
    "Pastor Alemán", "Pitbull Terrier", "Pomerania",
    "Pug (Carlino)", "Rottweiler", "San Bernardo",
    "Schnauzer Miniatura", "Shar Pei", "Shih Tzu",
    "Xoloitzcuintle", "Yorkshire Terrier"
]

AGE_GROUPS = {
    "0-1": "01",
    "1-4": "14",
    "4-8": "48",
    "8-13": "813"
}

rows = []

for breed in BREEDS:
    breed_no_spaces = breed.replace(" ", "").replace("(", "").replace(")", "")
    for age_range, age_code in AGE_GROUPS.items():
        age_folder = f"{age_range} anos"
        for i in range(1, IMAGES_PER_AGE + 1):
            filename = f"{breed_no_spaces}{age_code}_{i}.jpg"
            image_path = f"{BASE_PATH}/{breed}/{age_folder}/{filename}"
            rows.append([image_path, breed, age_range])

with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "breed", "age_range"])
    writer.writerows(rows)

print("CSV generado correctamente")
print(f"Total de imágenes listadas: {len(rows)}")
