from torchvision import transforms
from configs.config import IMAGE_SIZE


transform_train = transforms.Compose([
    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.85, 1.0),
    ),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(
        degrees=5,
        fill=0,
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        fill=0,
    ),

    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])


transform_test = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])
