import yaml
import redis
import io
from PIL import Image

class Config:
    config = yaml.load(open("config/config.yaml"), Loader=yaml.FullLoader)
    YOLO_MODEL = config["YOLO"]["detection_model"]
    CONFIDENCE = config["YOLO"]["confidence"]
    USE_TENSORRT = config["YOLO"]["use_tensorrt"]
    MAX_NEW_TOKENS = config["VLM"]["max_new_tokens"]
    QWEN_MODEL = config["VLM"]["model"]
    CLIP_MODELS = config["EMBEDDING"]["model"]


class RedisManager:
    def __init__(self, host="localhost", port=6379, db=0):
        self.host = host
        self.port = port
        self.db = db
        self.redis = redis.Redis(host=host, port=port, db=db)

    def set_value(self, key, value):
        self.redis.set(key, value)

    def get_value(self, key):
        return self.redis.get(key)


def image_encoding_using_pillow(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG",quality=100)
    return buffer.getvalue()

def image_decoding_using_pillow(encoded_image):
    decoded_image = Image.open(io.BytesIO(encoded_image))
    return decoded_image
