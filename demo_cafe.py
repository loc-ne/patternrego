import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import torchvision.models as models
import clip

# 1. Configuration
class Args:
    raf_path = "demo-checkpoint"
    gpu = 0
    
args = Args()
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

# 4. New Architecture (similar to f.py)
class Model(nn.Module):
    def __init__(self, device, num_classes=7):
        super(Model, self).__init__()
        
        # Load CLIP model
        self.clip_model, _ = clip.load("ViT-B/32", device=device) 
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # ResNet-50 Backbone
        res50 = models.resnet50(weights=None)
        self.features = nn.Sequential(*list(res50.children())[:-2])
        self.features2 = nn.Sequential(*list(res50.children())[-2:-1])
        
        # Reduction layer: reduce 2048 to 512 through mean operation
        self.reduction = nn.AvgPool1d(kernel_size=4, stride=4)
        
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x, phase='test'):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(x)
            
        x = self.features(x)
        x = self.features2(x)
        x = x.view(x.size(0), -1)    
        
        # Reduce dimension from 2048 to 512 using sliding window mean (non-overlapping)
        x = x.view(x.size(0), 1, -1) # (N, 1, 2048)
        x = self.reduction(x)        # (N, 1, 512)
        x = x.view(x.size(0), -1)    # (N, 512)
        
        # Fusion with CLIP features using Sigmoid Mask
        x = image_features * torch.sigmoid(x)
        out = self.fc(x)
        
        return out, out

# 5. Initialize and Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {device}")

model = Model(device=device, num_classes=7)

# Load checkpoint from demo-checkpoint
checkpoint_path = os.path.join('demo-checkpoint', 'ours_best.pth')
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    # Handle DataParallel prefix if necessary
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    print("Checkpoint loaded successfully.")
else:
    print(f"Warning: Checkpoint not found at {checkpoint_path}")

model.to(device)
model.eval()

# 6. Webcam Demo
cap = cv2.VideoCapture(0)
print("Đang mở Webcam... Nhấn 'q' để thoát.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        if face_img.size == 0:
            continue
            
        # Preprocess and Predict
        input_tensor = transform(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs, _ = model(input_tensor, phase='test')
            _, predicted = torch.max(outputs, 1)
            emotion = EMOTIONS[predicted.item()]

        # Draw results
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('Generalizable FER Demo (New Architecture)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

