import os

current_file_path = os.path.abspath(__file__)
base_dir = os.path.dirname(os.path.dirname(current_file_path))

DATA_FOLDER = os.path.join(base_dir, "data")
CKPT_FOLDER = os.path.join(base_dir, "checkpoints")
