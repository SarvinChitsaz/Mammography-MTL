import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from configs.config import IMAGE_SIZE

class CBISDDSMMultiTaskDataset(Dataset):
    def __init__(
        self,
        dataframe,
        transform=None,
        input_mode="cropped",
        resolve_image_path=None,
    ):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
        self.input_mode = input_mode
        self.resolve_image_path = resolve_image_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        if self.input_mode == "cropped":
            img_path = self.resolve_image_path(
                row["cropped image file path"]
            )
        else:
            raise ValueError(
                "Only cropped input_mode is used in this project."
            )

        img = cv2.imread(
            str(img_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if img is None:
            raise FileNotFoundError(
                f"Could not read image: {img_path}"
            )

        img = cv2.resize(
            img,
            (IMAGE_SIZE, IMAGE_SIZE),
        )

        img = cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2RGB,
        )

        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        lesion_label = int(row["lesion_label"])
        pathology_label = int(row["pathology_label"])

        return (
            img,
            lesion_label,
            pathology_label,
        )
