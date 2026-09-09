import yaml

class Config:
    config = yaml.load(open("config/config.yaml"), Loader=yaml.FullLoader)
    YOLO_MODEL = config["YOLO"]["detection_model"]
    CONFIDENCE = config["YOLO"]["confidence"]
    USE_TENSORRT = config["YOLO"]["use_tensorrt"]
    MAX_NEW_TOKENS = config["VLM"]["max_new_tokens"]
    QWEN_MODEL = config["VLM"]["model"]
