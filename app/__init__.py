"""SmearScan: research web app for cell detection / classification on blood-smear images.

Model weights come from the local Hugging Face / timm cache only (no downloads at run time).
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
