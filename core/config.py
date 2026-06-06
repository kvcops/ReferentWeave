import os
import logging
import colorama
from pathlib import Path

# Initialize colorama for colored logs on Windows
colorama.init()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "index"
UPLOAD_DIR = DATA_DIR / "uploads"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# Load .env file manually if it exists
env_path = BASE_DIR / ".env"
if env_path.exists():
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    os.environ[key] = val
    except Exception:
        pass

# API keys and Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
JINA_API_KEY = os.getenv("JINA_API_KEY", "jina_c84185dd898e4876bd0b2599d17ed9a7KEW8wFI-gSbyps6gdQfwa8gm_bJp")

# Model configuration
GEMINI_MODEL = "gemini-3.1-flash-lite"
JINA_EMBEDDING_MODEL = "jina-embeddings-v5-text-small"
EMBEDDING_DIM = 1024

# Search Parameters
TOP_K_RETRIEVAL = 50
TOP_K_GENERATION = 5
RRF_CONSTANT = 60
MAX_DOCLING_PAGES = 30

# Logger setup
class ColoredFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: colorama.Fore.CYAN,
        logging.INFO: colorama.Fore.GREEN,
        logging.WARNING: colorama.Fore.YELLOW,
        logging.ERROR: colorama.Fore.RED,
        logging.CRITICAL: colorama.Back.RED + colorama.Fore.WHITE
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, colorama.Fore.RESET)
        reset = colorama.Fore.RESET + colorama.Back.RESET
        record.levelname = f"{color}{record.levelname}{reset}"
        record.msg = f"{color}{record.msg}{reset}"
        return super().format(record)

def setup_logger():
    logger = logging.getLogger("ReferentWeave")
    logger.setLevel(logging.DEBUG)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = ColoredFormatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler for complete logging
        log_file = DATA_DIR / "referent_weave.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d] - %(message)s'
        )
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)
        
    return logger

logger = setup_logger()
logger.info(f"ReferentWeave system paths initialized. Base directory: {BASE_DIR}")
