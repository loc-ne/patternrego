import os
import cv2
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
import torchvision.models as models
import clip

class RafDataset(data.Dataset):
    def __init__(self, dataset_path, transform=None, crop_face=False, dataset_name='', split='test'):
        self.transform = transform
        self.crop_face = crop_face
        self.dataset_name = dataset_name
        self.split = split
        if self.crop_face:
            # Load face detector
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
        label_path = os.path.join(dataset_path, 'EmoLabel', 'list_patition_label.txt')
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found at {label_path}")
        
        dataset = pd.read_csv(label_path, sep=' ', header=None)
        
        # Filter based on split
        if self.split != 'all':
            if self.dataset_name == 'raf':
                mask = dataset[0].str.startswith(self.split)
                dataset = dataset[mask]
            elif self.dataset_name == 'sfew':
                if self.split == 'test':
                    mask = dataset[0].str.startswith('Val') | dataset[0].str.startswith('Test')
                else:
                    mask = dataset[0].str.startswith('Train')
                dataset = dataset[mask]
            elif self.dataset_name == 'mma':
                if self.split == 'test':
                    mask = dataset[0].str.startswith('mma_test') | dataset[0].str.startswith('mma_valid')
                else:
                    mask = dataset[0].str.startswith('mma_train')
                dataset = dataset[mask]

        self.file_paths = dataset.iloc[:, 0].values
        labels = dataset.iloc[:, 1].values
        
        if self.dataset_name == 'raf':
            # Map RAF-DB original labels (1-7) to Model labels (0-6)
            # RAF: 1: Surprise, 2: Fear, 3: Disgust, 4: Happiness, 5: Sadness, 6: Anger, 7: Neutral
            # Model: 0: Neutral, 1: Happy, 2: Surprise, 3: Sad, 4: Angry, 5: Disgust, 6: Fear
            raf_map = {1: 2, 2: 6, 3: 5, 4: 1, 5: 3, 6: 4, 7: 0}
            self.label = np.array([raf_map[lbl] if lbl in raf_map else lbl for lbl in labels])
        else:
            self.label = labels
            
        self.base_img_path = os.path.join(dataset_path, 'Image', 'aligned')

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_name = str(self.file_paths[idx])
        name_without_ext, ext_orig = os.path.splitext(img_name)
        
        # Try both base name, name + '_aligned', and name without '_aligned' if present
        candidates = [img_name, name_without_ext]
        if '_aligned' not in name_without_ext:
            candidates.append(name_without_ext + '_aligned')
        else:
            candidates.append(name_without_ext.replace('_aligned', ''))
            
        exts = ['', '.jpg', '.png', '.tiff', '.jpeg', '.tif']
        image = None
        
        # Try finding in self.base_img_path
        for candidate in candidates:
            for ext in exts:
                path = os.path.join(self.base_img_path, candidate + ext)
                if os.path.exists(path):
                    image = cv2.imread(path)
                    if image is not None:
                        break
            if image is not None:
                break
                
        # Fallback to absolute or relative path directly
        if image is None:
            for candidate in candidates:
                for ext in exts:
                    path = candidate + ext
                    if os.path.exists(path):
                        image = cv2.imread(path)
                        if image is not None:
                            break
                if image is not None:
                    break
                    
        if image is None:
            print(f"Warning: Image not found for index {idx}: {img_name}")
            return torch.zeros(3, 224, 224), self.label[idx], idx, torch.zeros(3, 224, 224)

        if self.crop_face:
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    # Crop face with a 10% margin
                    h_img, w_img, _ = image.shape
                    pad = int(w * 0.1)
                    y1 = max(0, y - pad)
                    y2 = min(h_img, y + h + pad)
                    x1 = max(0, x - pad)
                    x2 = min(w_img, x + w + pad)
                    image = image[y1:y2, x1:x2]
            except Exception as e:
                # Silently fallback to raw image if detection fails
                pass

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.label[idx]
        image_tensor = self.transform(img_rgb)
        img1 = transforms.RandomHorizontalFlip(p=1.0)(image_tensor)
        return image_tensor, label, idx, img1

class Model(nn.Module):
    def __init__(self, pretrained=True, num_classes=7, drop_rate=0):
        super(Model, self).__init__()
        
        # Load CLIP model inside to facilitate DataParallel
        self.clip_model, _ = clip.load("ViT-B/32", device='cpu') 
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Use ResNet-50 as backbone as specified by the author
        res50 = models.resnet50(pretrained=pretrained)
        self.drop_rate = drop_rate
        self.features = nn.Sequential(*list(res50.children())[:-2])
        self.features2 = nn.Sequential(*list(res50.children())[-2:-1])
        
        # Reduction layer: reduce 2048 to 512 through mean operation using sliding windows
        self.reduction = nn.AvgPool1d(kernel_size=4, stride=4)
        
        self.fc = nn.Linear(512, num_classes) # Reduced dimension is 512
        
        self.parm={}
        for name,parameters in self.fc.named_parameters():
            self.parm[name]=parameters
        
    def forward(self, x, targets=None, phase='test'):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(x)
            
        x = self.features(x)
        feat = x
        
        x = self.features2(x)
        x = x.view(x.size(0), -1)    
        
        # Reduce dimension from 2048 to 512 using sliding window mean (non-overlapping)
        x = x.view(x.size(0), 1, -1) # (N, 1, 2048)
        x = self.reduction(x) # (N, 1, 512)
        x = x.view(x.size(0), -1) # (N, 512)

        x = image_features * torch.sigmoid(x)
        out = self.fc(x)
        
        return out, out

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_checkpoint(model, path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at {path}")
    
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # Strip 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=True)
    print(f"Successfully loaded checkpoint from: {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ck', choices=['ck', 'jaffe', 'mma', 'raf', 'sfew', 'custom'], help='Dataset to test')
    parser.add_argument('--dataset_path', type=str, default='', help='Path to custom dataset directory if choice is custom')
    parser.add_argument('--checkpoint_path', type=str, 
                        default='/kaggle/input/datasets/nguynquclc/checkpoint-resnet50/ours_best_resnet50.pth', 
                        help='Path to the model checkpoint weights')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--crop_face', action='store_true', help='Use Haar Cascade to crop face from raw image')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'train', 'all'], help='Dataset split to evaluate')
    args = parser.parse_args()

    setup_seed(3407)

    # Determine dataset path
    code_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(code_dir)
    base_dir = parent_dir
    
    if args.dataset == 'ck':
        dataset_path = os.path.join(code_dir, 'ck-basic')
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(parent_dir, 'ck-basic')
            
    elif args.dataset == 'jaffe':
        dataset_path = os.path.join(code_dir, 'jaffe-basic')
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(parent_dir, 'jaffe-basic')
            
    elif args.dataset == 'mma':
        # Prioritize Kaggle path for MMA
        dataset_path = '/kaggle/input/datasets/nguynquclc/mma-basic/mma-basic'
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(code_dir, 'mma-basic')
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(parent_dir, 'mma-basic')
            
    elif args.dataset == 'raf':
        # Prioritize Kaggle path for RAF-DB
        dataset_path = '/kaggle/input/datasets/nguynquclc/raf-db/raf-basic'
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(code_dir, 'raf-basic')
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(parent_dir, 'raf-basic')
            
    elif args.dataset == 'sfew':
        # Prioritize Kaggle path for SFEW
        dataset_path = '/kaggle/input/datasets/nguynquclc/sfew2-0/sfew-basic'
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(code_dir, 'sfew-basic')
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(parent_dir, 'sfew-basic')
            
    else:
        dataset_path = args.dataset_path

    print(f"Testing on dataset: {args.dataset.upper()} (Path: {dataset_path}, Crop Face: {args.crop_face}, Split: {args.split})")

    eval_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    try:
        test_dataset = RafDataset(dataset_path=dataset_path, transform=eval_transforms, crop_face=args.crop_face, dataset_name=args.dataset, split=args.split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please run 'python reorganize_datasets.py' first to format the datasets.")
        return

    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Initialize model (ResNet50 based)
    model = Model(pretrained=False, num_classes=7)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Try loading the checkpoint
    try:
        load_checkpoint(model, args.checkpoint_path, device)
    except Exception as e:
        print(f"Error loading main checkpoint from {args.checkpoint_path}: {e}")
        # Try fallback to local ours_best.pth or ours_final.pth
        fallback_paths = [
            'ours_best.pth',
            'ours_final.pth',
            'code/ours_best.pth',
            'code/ours_final.pth',
            '/kaggle/input/datasets/nguynquclc/train-diemcong/ours_final.pth'
        ]
        loaded = False
        for path in fallback_paths:
            full_path = os.path.join(base_dir, path) if not os.path.isabs(path) else path
            if os.path.exists(full_path):
                try:
                    load_checkpoint(model, full_path, device)
                    loaded = True
                    break
                except Exception as ex:
                    print(f"Failed loading fallback checkpoint {full_path}: {ex}")
        if not loaded:
            print("Could not load any checkpoints. Exiting.")
            return

    model.to(device)

    with torch.no_grad():
        model.eval()
        correct_sum = 0
        data_num = 0

        for batch_i, (imgs1, labels, indexes, imgs2) in enumerate(test_loader):
            imgs1 = imgs1.to(device)
            labels = labels.to(device)

            outputs, _ = model(imgs1)

            _, predicts = torch.max(outputs, 1)
            correct_num = torch.eq(predicts, labels).sum()
            correct_sum += correct_num
            data_num += outputs.size(0)

        test_acc = correct_sum.float() / float(data_num)
        print(f"=========================================")
        print(f"Accuracy on {args.dataset.upper()} test set: {test_acc:.4f} ({correct_sum}/{data_num})")
        print(f"=========================================")

if __name__ == '__main__':
    main()