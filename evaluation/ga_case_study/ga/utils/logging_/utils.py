from datetime import datetime
import json
from typing import Tuple, Any


def duration_to_seconds(duration: str) -> float:
    hours = int(duration.split(":")[0])
    minutes = int(duration.split(":")[1])
    seconds = float(duration.split(":")[2])

    return seconds + minutes * 60 + hours * 60 * 60


def save_to_json(obj, path: str) -> None:
    with open(path, "w") as config_file:
        json.dump(obj, config_file)


def load_from_json(path: str) -> Any:
    with open(path, "r") as config_file:
        return json.load(config_file)


def get_timestamp() -> str:
    return str(datetime.now())
