import os
from pathlib import Path

import dotenv
import nltk

NLTK_DATA_PATH = os.path.join(Path(__file__).parents[2], "applications/nltk_data")
nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

dotenv.load_dotenv(os.path.join(Path(__file__).parents[1], ".env"))
