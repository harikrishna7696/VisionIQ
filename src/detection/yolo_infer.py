import json
import logging
import os
import uuid
from typing import Optional

import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from . import grounding_dino
from ..captioning.captions_qwen import Qwen_Model
from ..captioning.embedding_clip import ClipEmbeddings
from app.utils import Config, RedisManager, image_encoding_using_pillow, image_decoding_using_pillow, futuristic_ai_loader
from app.vector_db import LanceDB
from ..detection.grounding_dino import GroundingDino

# Module-level logger is preferable to using print() throughout the application.
logger = logging.getLogger(__name__)


class YOLOInfer:
    """
    Handles YOLO object detection/tracking, object cropping,
    image caption generation, and crop storage in Redis.

    The class supports both standard PyTorch YOLO models (.pt)
    and TensorRT engines (.engine).
    """

    def __init__(self, model_path: str,use_tensorrt: bool,confidence: float,
        caption_once_per_track: bool = True,):
        """
        Initialize YOLO inference, captioning, and Redis components.

        Args:
            model_path:
                Path to the YOLO .pt model.

            use_tensorrt:
                If True, use an existing TensorRT engine or export
                the PyTorch model to TensorRT.

            confidence:
                Minimum YOLO detection confidence.

            caption_once_per_track:
                If True, generate a Qwen caption once for each tracked
                object and reuse it while the same track ID exists.

                This can significantly improve performance because
                vision-language caption generation is typically much
                more expensive than object detection.

                Set this to False if captions must reflect changes in
                the object's appearance on every frame.
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"confidence must be between 0 and 1, got {confidence}"
            )

        self.use_tensorrt = use_tensorrt
        self.confidence = confidence
        self.caption_once_per_track = caption_once_per_track

        # Export/load the appropriate YOLO model.
        self.model_path = self.export_to_trt(model_path)

        # Redis is used to store encoded cropped images.
        self.redis = RedisManager()

        self.lanceDB = LanceDB()

        # Cache captions using YOLO track ID:
        # {
        #     track_id: "generated caption",
        #     ...
        # }
        #
        # This prevents repeated Qwen inference for the same tracked
        # object when caption_once_per_track=True.
        self._caption_cache: dict[int, list[str]] = {}

        self.data = []

    def run_yolo_inference(self,video_path:str):
        yolo = YOLO(self.model_path)
        video = cv2.VideoCapture(video_path)
        frame_id = 0

        if not video.isOpened():
            video.release()
            raise RuntimeError(f"Could not open video: {video_path}")
        data = []
        while True:
            # --------------------------------------------------
            # Read frame
            # --------------------------------------------------
            success, frame = video.read()

            if not success:
                break

            detection_placeholder = st.empty()
            frame_height, frame_width = frame.shape[:2]

            # Unique ID representing this frame's metadata.
            frame_uid = str(uuid.uuid4())

            total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_metadata = {
                "id": frame_uid,
                "video_path": video_path,
                "frame_id": frame_id,
                "cropped_images_ids": [],
                "detections": [],
                "frame_time": None
            }

            frame_id += 1

            # --------------------------------------------------
            # YOLO detection + tracking
            # --------------------------------------------------
            #
            # Passing confidence directly to YOLO means low
            # confidence detections can be discarded before our
            # Python processing loop.
            results = yolo.track(
                frame,
                persist=True,
                verbose=False,
                conf=self.confidence,
            )
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            if not results:
                continue

            result = results[0]
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                continue

            # --------------------------------------------------
            # Video timestamp
            # --------------------------------------------------
            timestamp_ms = video.get(cv2.CAP_PROP_POS_MSEC)
            timestamp = self._format_timestamp(timestamp_ms)

            logger.info("Processing frame=%d video_time=%s detections=%d",frame_id - 1, timestamp,len(boxes),)

            # --------------------------------------------------
            # Move YOLO results from GPU -> CPU once
            # --------------------------------------------------
            #
            # The original implementation repeatedly called
            # .cpu() inside the detection loop. Moving entire
            # tensors once is substantially cleaner and reduces
            # synchronization overhead.
            coordinates = boxes.xyxy.cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            class_ids = boxes.cls.int().cpu().tolist()

            if boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
            else:
                track_ids = [None] * len(boxes)

            # --------------------------------------------------
            # Process detections
            # --------------------------------------------------
            for (coordinates_xyxy,
                    confidence,
                    class_id,
                    track_id,
            ) in zip(coordinates, confidences, class_ids, track_ids):
                # This is mostly a safety check because YOLO has
                # already applied the confidence threshold above.
                if confidence < self.confidence:
                    continue

                x1, y1, x2, y2 = map(int, coordinates_xyxy)

                # Clamp bounding boxes to the frame boundaries.
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame_width, x2)
                y2 = min(frame_height, y2)

                # Ignore invalid/zero-area boxes.
                if x2 <= x1 or y2 <= y1:
                    continue
                class_name = result.names[class_id]
                frame_metadata["detections"].append([class_name, confidence, x1, y1, x2, y2])
                # Clear old message and write new one

            frame_metadata["frame_time"] = timestamp
            print(f"frame: {frame_id}/{total_frame}")
            self.redis.set_value(frame_uid, image_encoding_using_pillow(frame))
            self.redis.set_value(frame_uid+"_data", json.dumps(frame_metadata))
            data.append(frame_metadata)
            # detection_placeholder.empty()


        video.release()
        yolo = None
        return data

    def run_qwen_inference(self, prompt, data):
        with st.spinner("Loading the Qwen model for Captioning"):
            qwen_model = Qwen_Model(Config.QWEN_MODEL, Config.MAX_NEW_TOKENS)
            clip_model = ClipEmbeddings(Config.CLIP_MODELS)
        with st.spinner("Running Captions for the images"):
            for i, dt in enumerate(data):
                try:
                    image_uid = dt["id"]
                    image = image_decoding_using_pillow(self.redis.get_value(image_uid))
                    detections = dt["detections"]
                    child_uids = []
                    for det in detections:
                        cr_id = str(uuid.uuid4())
                        child_uids.append(cr_id)
                        class_name = det[0]
                        conf = det[1]
                        crop_image = image.crop(det[2:])
                        caption = qwen_model.caption(crop_image, prompt)
                        image_embeddings = clip_model.img_embeddings(crop_image)
                        text_embeddings = clip_model.text_embeddings(caption)
                        crop_image_metadata = {"id": cr_id,
                                               "label": class_name,
                                               "caption": caption,
                                               "confidence": conf,
                                               "parent_id": image_uid,
                                               "image_embeddings": image_embeddings,
                                               "text_embeddings": text_embeddings}
                        self.lanceDB.insert([crop_image_metadata])
                    print(f"Processed frame for captioning: {i+1}/{len(data)}")
                except Exception as e:
                    print(e)
                    print(dt)
        qwen_model = None
        return clip_model

    def run_clip_inference(self):
        pass

    def export_to_trt(self, model_path: str) -> str:
        """
        Return the model path to use for YOLO inference.

        If TensorRT is enabled:
            1. Check whether the corresponding .engine already exists.
            2. If it exists, reuse it.
            3. Otherwise export the .pt model to TensorRT.

        Args:
            model_path:
                Path to the YOLO PyTorch model.

        Returns:
            Path to either the original model or TensorRT engine.

        Raises:
            FileNotFoundError:
                If the supplied model does not exist.
        """
        if not os.path.exists(model_path):
            print(f"YOLO model not found: {model_path}")

        # No conversion is required when TensorRT is disabled.
        if not self.use_tensorrt:
            return model_path

        # Example:
        # yolov8.pt -> yolov8.engine
        engine_path = os.path.splitext(model_path)[0] + ".engine"

        # Avoid exporting the model every time the application starts.
        if os.path.exists(engine_path):
            logger.info("Using existing TensorRT engine: %s", engine_path)
            return engine_path

        logger.info("TensorRT engine not found. Exporting %s", model_path)

        st.info(
            "Converting the YOLO model to TensorRT. "
            "This only needs to be done once."
        )

        model = YOLO(model_path)

        # ultralytics returns the exported model path.
        exported_path = model.export(
            format="engine",
            device=0,
        )

        exported_path = str(exported_path)

        logger.info("TensorRT model exported to: %s", exported_path)
        st.success(f"TensorRT model exported to: {exported_path}")

        return exported_path

    @staticmethod
    def _format_timestamp(milliseconds: float) -> str:
        """
        Convert a video timestamp in milliseconds to HH:MM:SS.

        Args:
            milliseconds:
                Timestamp returned by OpenCV CAP_PROP_POS_MSEC.

        Returns:
            Timestamp formatted as HH:MM:SS.
        """
        total_seconds = int(milliseconds / 1000)

        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _get_caption(
        self,
        prompt_text: str,
        crop: Image.Image,
        track_id: Optional[int],
    ) -> str:
        """
        Generate or retrieve a caption for a detected object.

        When caption caching is enabled, tracked objects are captioned
        only once. This avoids repeatedly calling the expensive Qwen
        vision-language model for the same object.

        Args:
            crop:
                PIL image containing the detected object.

            track_id:
                YOLO tracker ID. None when the object has not received
                a tracker ID.

        Returns:
            Generated image caption.
        """

        # Qwen inference is one of the most computationally expensive
        # operations in the processing pipeline.
        caption = self.qwen_model.caption(crop, (prompt_text))
        if caption and track_id is not None:
            if track_id not in self._caption_cache:
                self._caption_cache[track_id] = [caption]
            if caption not in self._caption_cache[track_id]:
                self._caption_cache[track_id].append(caption)
            return self._caption_cache[track_id][-1]
        if self.caption_once_per_track and track_id is not None:
            self._caption_cache[track_id] = caption

        return caption

    def read_video_get_detections(self, video_path: str, ui, prompt_text):
        """
        Process a video frame-by-frame using YOLO tracking.

        For every valid detection:
            1. Obtain the bounding box and object class.
            2. Crop the detected object.
            3. Convert the crop from OpenCV BGR to RGB.
            4. Generate an image caption.
            5. Encode and store the crop in Redis.
            6. Display the crop and caption in Streamlit.

        Args:
            video_path:
                Path to the input video.

            ui:
                Streamlit-compatible object used for displaying images.
                Usually pass ``st`` when calling this method.

        Raises:
            RuntimeError:
                If OpenCV cannot open the supplied video.
        """
        # Do not accidentally reuse captions from a previously
        # processed video.
        self._caption_cache.clear()
        # loader = futuristic_ai_loader("Detecting the objects")
        data = self.run_yolo_inference(video_path=video_path)
        # loader.empty()
        st.success("Detections Completed")
        # --------------------------------------------------
        # Caption generation
        # --------------------------------------------------

        # loader = futuristic_ai_loader("Running captioning and embeddings on images")
        with st.spinner("Running the Captioning"):
            self.clip_model = self.run_qwen_inference(prompt_text, data)
        st.success("Clip Model Completed")


    def fetch_frames_match_to_prompt_text(self, prompt, st):
        text_embedding = self.clip_model.text_embeddings(prompt)
        results = self.lanceDB.table.search(
            text_embedding,
            vector_column_name="text_embeddings",
        ).limit(5).to_list()
        self.clip_model = None
        st.write(results)
        result = results[0]
        caption = result["caption"]
        parent_id = result["parent_id"]
        redis_value = self.redis.get_value(parent_id)
        image = image_decoding_using_pillow(redis_value)
        grounding_dino = GroundingDino(Config.GROUNDING_DINO)
        image = grounding_dino.detect(image, caption)
        return image, caption
