from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

AI_MODEL_VERSION = "yolov8-waste-v3.1"
MIN_AI_CONFIDENCE = 0.60
INFERENCE_LATENCY_P95_TARGET_MS = 800

DEFAULT_REGISTRY_PATH = BASE_DIR / "data" / "containers_registry.json"
DEFAULT_AUDIT_LOG_PATH = BASE_DIR / "runtime" / "audit_log.jsonl"

ORGANIC_TACO_CLASSES = {
    "food_waste",
    "organic_matter",
    "biohazard_waste",
    "decomposed_material",
    "food_packaging_soiled",
}

ORGANIC_CLASS_ALIASES = {
    "food_waste": "food_waste",
    "organic_matter": "organic_matter",
    "biohazard_waste": "biohazard_waste",
    "decomposed_material": "decomposed_material",
    "food_packaging_soiled": "food_packaging_soiled",
    "disposable_food_container": "food_packaging_soiled",
    "foam_food_container": "food_packaging_soiled",
    "meal_carton": "food_packaging_soiled",
    "pizza_box": "food_packaging_soiled",

    # COCO food mapping aliases for testing
    "apple": "organic_matter",
    "banana": "organic_matter",
    "orange": "organic_matter",
    "broccoli": "organic_matter",
    "carrot": "organic_matter",
    "hot_dog": "food_waste",
    "pizza": "food_waste",
    "donut": "food_waste",
    "cake": "food_waste",
    "sandwich": "food_waste",
    "wine_glass": "food_packaging_soiled",
}

TRASH_CLASSES = {
    # TACO classes
    "trash", "waste", "garbage", "litter", "refuse",
    "food_waste", "organic_matter", "biohazard_waste", "decomposed_material", "food_packaging_soiled",
    "aluminium_foil", "plastic_bottle", "glass_bottle", "can", "paper", "cardboard", "carton",
    "plastic_bag", "disposable_food_container", "foam_food_container", "meal_carton", "pizza_box",
    # Waste model classes
    "glass", "metal", "plastic",
    # COCO classes mapped for testing
    "bottle", "cup", "wine_glass", "bowl", "backpack", "handbag", "umbrella",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot_dog", "pizza", "donut", "cake"
}

