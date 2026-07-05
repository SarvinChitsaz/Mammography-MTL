import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from configs.config import IMAGE_SIZE, BATCH_SIZE
from data.prepare_data import resolve_image_path
from data.transforms import transform_train, transform_test


class CBISDDSMMultiTaskDataset(Dataset):
    def __init__(
        self,
        dataframe,
        transform=None,
        input_mode="cropped",
        uid_to_path=None,
    ):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
        self.input_mode = input_mode
        self.uid_to_path = uid_to_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        if self.input_mode == "cropped":
            img_path = resolve_image_path(
                row["cropped image file path"],
                self.uid_to_path,
            )
        else:
            raise ValueError("Only cropped input_mode is used in this project.")

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        lesion_label = int(row["lesion_label"])
        pathology_label = int(row["pathology_label"])

        return img, lesion_label, pathology_label


def create_dataloaders(
    train_df,
    val_df,
    test_df,
    uid_to_path,
    input_mode="cropped",
):
    train_dataset = CBISDDSMMultiTaskDataset(
        train_df,
        transform=transform_train,
        input_mode=input_mode,
        uid_to_path=uid_to_path,
    )

    val_dataset = CBISDDSMMultiTaskDataset(
        val_df,
        transform=transform_test,
        input_mode=input_mode,
        uid_to_path=uid_to_path,
    )

    test_dataset = CBISDDSMMultiTaskDataset(
        test_df,
        transform=transform_test,
        input_mode=input_mode,
        uid_to_path=uid_to_path,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset,
    )
