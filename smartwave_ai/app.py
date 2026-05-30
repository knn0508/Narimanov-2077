import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp"

from dotenv import load_dotenv
load_dotenv()

from smartwave_ai.visual_analysis.api import create_app


app = create_app()

