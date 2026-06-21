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
    def __init__(self, dataset_path, transform=None):
        self.transform = transform
        label_path = os.path.join(dataset_path, 'EmoLabel', 'list_patition_label.txt')
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found at {label_path}")
        
        dataset = pd.read_csv(label_path, sep=' ', header=None)
        self.file_paths = dataset.iloc[:, 0].values
        self.label = dataset.iloc[:, 1].values
        self.base_img_path = os.path.join(dataset_path, 'Image', 'aligned')

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_name = str(self.file_paths[idx])
        exts = ['', '.jpg', '.png', '.tiff', '.jpeg']
        image = None
        for ext in exts:
            path = os.path.join(self.base_img_path, img_name + ext)
            if os.path.exists(path):
                image = cv2.imread(path)
                if image is not None:
                    break
        
        if image is None:
            # Fallback if the path recorded in txt is already absolute or has extension
            for ext in exts:
                path = img_name + ext
                if os.path.exists(path):
                    image = cv2.imread(path)
                    if image is not None:
                        break
        
        if image is None:
            print(f"Warning: Image not found for index {idx}: {img_name}")
            return torch.zeros(3, 224, 224), self.label[idx], idx, torch.zeros(3, 224, 224)

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
    parser.add_argument('--dataset', type=str, default='ck', choices=['ck', 'jaffe', 'mma', 'custom'], help='Dataset to test')
    parser.add_argument('--dataset_path', type=str, default='', help='Path to custom dataset directory if choice is custom')
    parser.add_argument('--checkpoint_path', type=str, 
                        default='/kaggle/input/datasets/nguynquclc/checkpoint-resnet50/ours_best_resnet50.pth', 
                        help='Path to the model checkpoint weights')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    args = parser.parse_args()

    setup_seed(3407)

    # Determine dataset path
    code_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(code_dir)
    if args.dataset == 'ck':
        dataset_path = os.path.join(code_dir, 'ck-basic')
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(parent_dir, 'ck-basic')
    elif args.dataset == 'jaffe':
        dataset_path = os.path.join(code_dir, 'jaffe-basic')
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(parent_dir, 'jaffe-basic')
    elif args.dataset == 'mma':
        dataset_path = os.path.join(code_dir, 'mma-basic')
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(parent_dir, 'mma-basic')
    else:
        dataset_path = args.dataset_path

    print(f"Testing on dataset: {args.dataset.upper()} (Path: {dataset_path})")

    eval_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    try:
        test_dataset = RafDataset(dataset_path=dataset_path, transform=eval_transforms)
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