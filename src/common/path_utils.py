
from datetime import datetime
import os
import string

def resolve_path(absolute_from_root: string) -> string:
    relative_from_root = f".{absolute_from_root}"
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), relative_from_root)

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d-%H-%M")