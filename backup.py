import cv2
import numpy as np
import onnxruntime as ort

race_labels = ["White", "Black", "Latino", "East Asian", "Southeast Asian", "Indian", "Middle Eastern"]
gender_labels = ["Male", "Female"]
age_labels = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]

# Load ONNX with TensorRT fallback
providers = [
    ("TensorrtExecutionProvider", {"trt_max_workspace_size": 1 << 30}),
    "CUDAExecutionProvider",
    "CPUExecutionProvider"
]

session = ort.InferenceSession("fairface_res34.onnx", providers=providers)

def preprocess(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    image = image.transpose(2, 0, 1)  # HWC to CHW
    return image[np.newaxis, :].astype(np.float32)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Webcam not found")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess(rgb)

    output = session.run(None, {"input": input_tensor})[0][0]

    race = race_labels[np.argmax(output[:7])]
    gender = gender_labels[np.argmax(output[7:9])]
    age = age_labels[np.argmax(output[9:])]

    label = f"{gender}, {race}, {age}"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("FairFace (ONNXRuntime + TensorRT)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
