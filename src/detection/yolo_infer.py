import logging
import os
import uuid
from typing import Optional

import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from ..captioning.captions_qwen import Qwen_Model
from app.utils import Config, RedisManager, image_encoding_using_pillow


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
        self.yolo = YOLO(self.model_path)

        # Vision-language model used for generating captions.
        self.qwen_model = Qwen_Model(
            Config.QWEN_MODEL,
            Config.MAX_NEW_TOKENS,
        )

        # Redis is used to store encoded cropped images.
        self.redis = RedisManager()

        # Cache captions using YOLO track ID:
        # {
        #     track_id: "generated caption",
        #     ...
        # }
        #
        # This prevents repeated Qwen inference for the same tracked
        # object when caption_once_per_track=True.
        self._caption_cache: dict[int, list[str]] = {}

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
            raise FileNotFoundError(f"YOLO model not found: {model_path}")

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
        caption = self.qwen_model.caption(crop)
        if caption and track_id is not None:
            if track_id not in self._caption_cache:
                self._caption_cache[track_id] = [caption]
            if caption not in self._caption_cache[track_id]:
                self._caption_cache[track_id].append(caption)
            return self._caption_cache[track_id][-1]
        if self.caption_once_per_track and track_id is not None:
            self._caption_cache[track_id] = caption

        return caption

    def read_video_get_detections(self, video_path: str, ui) -> None:
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
        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            video.release()
            raise RuntimeError(f"Could not open video: {video_path}")

        # Do not accidentally reuse captions from a previously
        # processed video.
        self._caption_cache.clear()

        frame_id = 0

        try:
            while True:
                # --------------------------------------------------
                # Read frame
                # --------------------------------------------------
                success, frame = video.read()

                if not success:
                    break

                frame_height, frame_width = frame.shape[:2]

                # Unique ID representing this frame's metadata.
                frame_uid = str(uuid.uuid4())

                frame_metadata = {
                    "id": frame_uid,
                    "video_path": video_path,
                    "frame_id": frame_id,
                    "detections": [],
                    "captions": [],
                    "cropped_images_ids": [],
                }

                frame_id += 1

                # --------------------------------------------------
                # YOLO detection + tracking
                # --------------------------------------------------
                #
                # Passing confidence directly to YOLO means low
                # confidence detections can be discarded before our
                # Python processing loop.
                results = self.yolo.track(
                    frame,
                    persist=True,
                    verbose=False,
                    conf=self.confidence,
                )

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

                logger.debug(
                    "Processing frame=%d video_time=%s detections=%d",
                    frame_id - 1,
                    timestamp,
                    len(boxes),
                )

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
                for (
                    coordinates_xyxy,
                    confidence,
                    class_id,
                    track_id,
                ) in zip(
                    coordinates,
                    confidences,
                    class_ids,
                    track_ids,
                ):
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

                    # --------------------------------------------------
                    # Crop detected object
                    # --------------------------------------------------
                    crop_bgr = frame[y1:y2, x1:x2]

                    if crop_bgr.size == 0:
                        continue

                    # OpenCV uses BGR whereas PIL/Qwen expects RGB.
                    crop_rgb = cv2.cvtColor(
                        crop_bgr,
                        cv2.COLOR_BGR2RGB,
                    )

                    crop_image = Image.fromarray(crop_rgb)

                    class_name = result.names[class_id]

                    # --------------------------------------------------
                    # Caption generation
                    # --------------------------------------------------
                    caption = self._get_caption(
                        crop=crop_image,
                        track_id=track_id,
                    )

                    # --------------------------------------------------
                    # Store metadata
                    # --------------------------------------------------
                    frame_metadata["detections"].append(
                        [
                            class_name,
                            x1,
                            y1,
                            x2,
                            y2,
                            float(confidence),
                            track_id,
                        ]
                    )

                    frame_metadata["captions"].append(caption)

                    # --------------------------------------------------
                    # Store crop in Redis
                    # --------------------------------------------------
                    crop_id = str(uuid.uuid4())

                    frame_metadata["cropped_images_ids"].append(
                        crop_id
                    )

                    encoded_crop = image_encoding_using_pillow(
                        crop_image
                    )

                    self.redis.set_value(
                        crop_id,
                        encoded_crop,
                    )

                    # --------------------------------------------------
                    # Streamlit output
                    # --------------------------------------------------
                    ui.image(
                        crop_image,
                        caption=caption,
                        use_container_width=True,
                    )

                # IMPORTANT:
                # frame_metadata currently exists only for this
                # iteration. Persist it here if the metadata is needed.
                #
                # Example possibilities:
                #
                # self.redis.set_value(frame_uid, ...)
                #
                # or send it to another database/message queue.
                #
                # Avoid accumulating every frame in a Python list for
                # long videos because memory usage can become very high.

        finally:
            # Ensure the video handle is always closed, even when an
            # inference/captioning exception occurs.
            video.release()

            logger.info(
                "Finished processing video '%s'. Frames read: %d",
                video_path,
                frame_id,
            )