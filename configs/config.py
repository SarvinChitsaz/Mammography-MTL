from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "dataset"
JPEG_DIR = DATA_DIR / "jpeg"
CSV_DIR = DATA_DIR / "csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = PROJECT_ROOT / "checkpoints"
GRADCAM_DIR = FIGURE_DIR / "gradcam"

LESION_CLASS_NAMES = ["Calcification", "Mass"]
PATHOLOGY_CLASS_NAMES = ["Benign", "Malignant"]

NUM_LESION_CLASSES = 2
NUM_PATHOLOGY_CLASSES = 2

MODEL_NAMES = ["resnet18", "densenet121", "efficientnet_b0", "mobilenet_v3_small"]
INPUT_MODES = ["cropped"]

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 10

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

SEED = 42
DROPOUT_RATE = 0.2
EARLY_STOPPING_PATIENCE = 5

LAMBDA_LESION = 1.0
LAMBDA_PATHOLOGY = 1.0
