import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import torchvision.models as models
import clip

# 1. Configuration
EMOTIONS = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Anger', 'Disgust', 'Fear']

# 2. Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 3. Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. Model Architecture (matches f.py)
class Model(nn.Module):
    def __init__(self, device, num_classes=7):
        super(Model, self).__init__()
        self.clip_model, _ = clip.load("ViT-B/32", device=device) 
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        res50 = models.resnet50(weights=None)
        self.features = nn.Sequential(*list(res50.children())[:-2])
        self.features2 = nn.Sequential(*list(res50.children())[-2:-1])
        self.reduction = nn.AvgPool1d(kernel_size=4, stride=4)
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x, phase='test'):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(x)
        x = self.features(x)
        x = self.features2(x)
        x = x.view(x.size(0), -1)    
        x = x.view(x.size(0), 1, -1)
        x = self.reduction(x)
        x = x.view(x.size(0), -1)
        x = image_features * torch.sigmoid(x)
        out = self.fc(x)
        return out, out

# 5. Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = Model(device=device, num_classes=7)
checkpoint_path = os.path.join('demo-checkpoint', 'ours_best.pth')

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    print("Model loaded successfully.")
else:
    print(f"Error: Checkpoint not found at {checkpoint_path}")
    exit()

model.to(device)
model.eval()

# 6. Run Inference on Image
image_path = 'test.jpg'
if not os.path.exists(image_path):
    print(f"Error: Image {image_path} not found.")
    exit()

frame = cv2.imread(image_path)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

if len(faces) == 0:
    print("No faces detected in the image.")
else:
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        input_tensor = transform(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs, _ = model(input_tensor, phase='test')
            _, predicted = torch.max(outputs, 1)
            emotion = EMOTIONS[predicted.item()]
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()

        # Draw result
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{emotion} ({confidence:.2f})", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        print(f"Detected: {emotion} with confidence {confidence:.2f}")

# Save result
output_path = 'result_test.jpg'
cv2.imwrite(output_path, frame)
print(f"Result saved to {output_path}")
