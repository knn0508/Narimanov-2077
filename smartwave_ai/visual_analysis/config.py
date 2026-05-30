from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

AI_MODEL_VERSION = "yolov8-waste-v3.1"
MIN_AI_CONFIDENCE = 0.60
INFERENCE_LATENCY_P95_TARGET_MS = 800

DEFAULT_REGISTRY_PATH = BASE_DIR / "data" / "containers_registry.json"
DEFAULT_AUDIT_LOG_PATH = BASE_DIR / "runtime" / "audit_log.jsonl"

# Classes expected in MIXED containers (general household/street waste)
MIXED_TACO_CLASSES = {
    "trash",
    "waste",
    "garbage",
    "litter",
    "food_waste",
    "food_packaging_soiled",
    "paper",
    "cardboard",
    "carton",
    "can",
    "glass_bottle",
    "aluminium_foil",
}

MIXED_CLASS_ALIASES = {
    "food_waste": "food_waste",
    "food_packaging_soiled": "food_packaging_soiled",
    "disposable_food_container": "food_packaging_soiled",
    "foam_food_container": "food_packaging_soiled",
    "meal_carton": "food_packaging_soiled",
    "pizza_box": "food_packaging_soiled",
    "paper": "paper",
    "cardboard": "cardboard",
    "carton": "carton",
    "can": "can",
    "glass_bottle": "glass_bottle",
    "aluminium_foil": "aluminium_foil",
    # COCO aliases
    "bottle": "glass_bottle",
    "cup": "food_packaging_soiled",
    "bowl": "food_packaging_soiled",
    "banana": "food_waste",
    "apple": "food_waste",
    "sandwich": "food_waste",
    "pizza": "food_waste",
    "donut": "food_waste",
    "cake": "food_waste",
}

# Classes expected in PLASTIC containers (plastic-specific waste)
PLASTIC_TACO_CLASSES = {
    "plastic_bottle",
    "plastic_bag",
    "plastic_container",
    "plastic",
    "plastic_lid",
    "plastic_straw",
    "plastic_film",
    "styrofoam",
    "foam_food_container",
}

PLASTIC_CLASS_ALIASES = {
    "plastic_bottle": "plastic_bottle",
    "plastic_bag": "plastic_bag",
    "plastic_container": "plastic_container",
    "plastic": "plastic",
    "plastic_lid": "plastic_lid",
    "plastic_straw": "plastic_straw",
    "plastic_film": "plastic_film",
    "styrofoam": "styrofoam",
    "foam_food_container": "foam_food_container",
    # COCO aliases
    "bottle": "plastic_bottle",
    "cup": "plastic_container",
    "backpack": "plastic_bag",
    "handbag": "plastic_bag",
}

TRASH_CLASSES = (
    MIXED_TACO_CLASSES
    | PLASTIC_TACO_CLASSES
    | {
        # additional COCO classes useful for mask area calculation
        "bottle", "cup", "wine_glass", "bowl", "backpack", "handbag", "umbrella",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
        "hot_dog", "pizza", "donut", "cake",
        # raw labels from waste-specific models
        "glass", "metal",
    }
)


