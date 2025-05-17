import cv2
import torch
import numpy as np
from torchvision import transforms, models
import torch.nn as nn
import logging
import time
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def load_model(path, num_classes=18):
    try:
        model = models.resnet34(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device).eval()
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

model = load_model("fair_face_models/res34_fair_align_multi_7_20190809.pt")

race_labels = ["White", "Black", "Latino", "East Asian", "Southeast Asian", "Indian", "Middle Eastern"]
gender_labels = ["Male", "Female"]
age_labels = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Failed to open webcam")
logger.info("Webcam opened successfully")

frame_count = 0
torch.set_grad_enabled(False)

true_genders, pred_genders = [], []
true_races, pred_races = [], []
true_ages, pred_ages = [], []

frame_labels = [
    (0, 5, 2),  # Male, Indian, 10-19
    (1, 5, 3),  # Female, Indian, 20-29
    (0, 5, 4),  # Male, Indian, 30-39
    (1, 5, 5),  # Female, Indian, 40-49
    (0, 5, 6),  # Male, Indian, 50-59
    (1, 5, 7),  # Female, Indian, 60-69
    (0, 5, 3),  # Male, Indian, 20-29
    (1, 5, 2),  # Female, Indian, 10-19
    (0, 5, 8),  # Male, Indian, 70+
    (1, 5, 1),  # Female, Indian, 3-9
]
label_index = 0

while True:
    t0 = time.time()
    ret, frame = cap.read()
    if not ret:
        logger.warning("Failed to grab frame")
        continue

    frame_count += 1
    try:
        detect_start = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        detect_end = time.time()
        infer_time = 0

        for (x, y, w, h) in faces:
            face_img = rgb[y:y+h, x:x+w]
            if face_img.size == 0:
                continue

            face_resized = cv2.resize(face_img, (224, 224))
            input_tensor = preprocess(face_resized).unsqueeze(0).to(device)

            infer_start = time.time()
            output = model(input_tensor).cpu().numpy().squeeze()
            infer_end = time.time()

            infer_time += infer_end - infer_start

            pred_race_idx = np.argmax(output[:7])
            pred_gender_idx = np.argmax(output[7:9])
            pred_age_idx = np.argmax(output[9:])

            label = f"{gender_labels[pred_gender_idx]}, {race_labels[pred_race_idx]}, {age_labels[pred_age_idx]}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 100), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if label_index < len(frame_labels):
                # Store predictions only if we have true labels
                pred_genders.append(pred_gender_idx)
                pred_races.append(pred_race_idx)
                pred_ages.append(pred_age_idx)

                tg, tr, ta = frame_labels[label_index]
                true_genders.append(tg)
                true_races.append(tr)
                true_ages.append(ta)
                label_index += 1


        t1 = time.time()
        logger.info(f"Frame {frame_count}: detect={detect_end - detect_start:.3f}s | infer={infer_time:.3f}s | total={t1 - t0:.3f}s")

        fps = 1 / (t1 - t0)
        cv2.putText(frame, f"{fps:.2f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("FairFace Real-Time (Fast)", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            logger.info("Exit command received")
            break

    except Exception as e:
        logger.error(f"Error during frame processing: {e}")
        break

cap.release()
cv2.destroyAllWindows()

from sklearn.metrics import precision_score, recall_score, f1_score

print("\nGender classification:")
print("Precision:", precision_score(true_genders, pred_genders, average='weighted', zero_division=0))
print("Recall:", recall_score(true_genders, pred_genders, average='weighted', zero_division=0))
print("F1 Score:", f1_score(true_genders, pred_genders, average='weighted', zero_division=0))

print("\nRace classification:")
print("Precision:", precision_score(true_races, pred_races, average='weighted', zero_division=0))
print("Recall:", recall_score(true_races, pred_races, average='weighted', zero_division=0))
print("F1 Score:", f1_score(true_races, pred_races, average='weighted', zero_division=0))

print("\nAge classification:")
print("Precision:", precision_score(true_ages, pred_ages, average='weighted', zero_division=0))
print("Recall:", recall_score(true_ages, pred_ages, average='weighted', zero_division=0))
print("F1 Score:", f1_score(true_ages, pred_ages, average='weighted', zero_division=0))

sns.heatmap(confusion_matrix(true_genders, pred_genders), annot=True, fmt="d", xticklabels=gender_labels, yticklabels=gender_labels)
plt.title("Gender Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()