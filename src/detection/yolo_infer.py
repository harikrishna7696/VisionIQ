import json
import logging
import os
import uuid
from typing import Optional

import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from ..captioning.captions_qwen import Qwen_Model
from ..captioning.embedding_clip import ClipEmbeddings
from app.utils import Config, RedisManager, image_encoding_using_pillow, image_decoding_using_pillow
from app.vector_db import LanceDB
from ..detection.grounding_dino import GroundingDino

logger = logging.getLogger(__name__)


class YOLOInfer:
    """
    Handles YOLO object detection/tracking, object cropping,
    image caption generation, and crop storage in Redis.
    """

    def __init__(self, model_path: str, use_tensorrt: bool, confidence: float, caption_once_per_track: bool = True,
                 temporal=False):
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be between 0 and 1, got {confidence}")

        self.use_tensorrt = use_tensorrt
        self.confidence = confidence
        self.caption_once_per_track = caption_once_per_track
        self.model_path = self.export_to_trt(model_path)
        self.redis = RedisManager()
        self.lanceDB = LanceDB()
        self._caption_cache: dict[int, list[str]] = {}
        self.data = []
        self.temporal = temporal

    @staticmethod
    def _draw_tracking_box(frame, x1, y1, x2, y2, class_name, track_id):
        """Draw bbox and class_name: track_id in the bbox center."""

        # Draw rectangle first so it stays behind the text.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        text = f"{class_name}: {track_id}" if track_id is not None else f"{class_name}: N/A"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness, padding = 0.7, 2, 5
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        tx, ty = cx - tw // 2, cy + th // 2
        h, w = frame.shape[:2]

        tx = max(padding, min(tx, w - tw - padding))
        ty = max(th + padding, min(ty, h - baseline - padding))

        cv2.rectangle(frame, (tx - padding, ty - th - padding), (tx + tw + padding, ty + baseline + padding), (0, 0, 0), -1)
        cv2.putText(frame, text, (tx, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def run_yolo_inference(self, video_path: str, output_video_path: Optional[str] = None):
        """
        Run YOLO detection/tracking.

        temporal=False:
            Normal detection. Stores frame metadata in Redis.

        temporal=True:
            Tracks objects, draws class_name: track_id in the bbox
            center, and saves the annotated video.
        """

        if self.temporal and not output_video_path:
            raise ValueError("output_video_path must be provided when temporal=True")

        yolo = YOLO(self.model_path)
        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            video.release()
            raise RuntimeError(f"Could not open video: {video_path}")

        frame_id = 0
        data = []

        fps = video.get(cv2.CAP_PROP_FPS) or 25.0
        total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_writer = None

        if self.temporal:
            output_dir = os.path.dirname(os.path.abspath(output_video_path))
            os.makedirs(output_dir, exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            save_path = "tracked_"+video_path.split("/")[-1]
            output_video_path = os.path.join(output_video_path, save_path)
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

            if not video_writer.isOpened():
                video.release()
                raise RuntimeError(f"Could not create output video: {output_video_path}")

            logger.info("Saving tracked video to: %s", output_video_path)

        detection_placeholder = st.empty()

        d = {}
        try:
            while True:
                success, frame = video.read()
                if not success:
                    break

                frame_uid = str(uuid.uuid4())
                current_frame_id = frame_id
                frame_id += 1

                frame_metadata = {
                    "id": frame_uid,
                    "video_path": video_path,
                    "frame_id": current_frame_id,
                    "cropped_images_ids": [],
                    "detections": [],
                    "frame_time": None,
                }

                # YOLO tracking.
                results = yolo.track(frame, persist=True, verbose=False, conf=self.confidence)

                if not results:
                    if self.temporal:
                        video_writer.write(frame)
                    continue

                result = results[0]
                boxes = result.boxes

                if boxes is None or len(boxes) == 0:
                    if self.temporal:
                        video_writer.write(frame)
                    continue

                timestamp = self._format_timestamp(video.get(cv2.CAP_PROP_POS_MSEC))
                frame_metadata["frame_time"] = timestamp

                coordinates = boxes.xyxy.cpu().tolist()
                confidences = boxes.conf.cpu().tolist()
                class_ids = boxes.cls.int().cpu().tolist()
                track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)

                logger.info(
                    "Processing frame=%d video_time=%s detections=%d",
                    current_frame_id,
                    timestamp,
                    len(boxes),
                )

                for coordinates_xyxy, confidence, class_id, track_id in zip(
                    coordinates, confidences, class_ids, track_ids
                ):
                    if confidence < self.confidence:
                        continue

                    x1, y1, x2, y2 = map(int, coordinates_xyxy)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame_width, x2), min(frame_height, y2)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    class_name = result.names[class_id]

                    if self.temporal:
                        self._draw_tracking_box(frame, x1, y1, x2, y2, class_name, track_id)
                    else:
                        frame_metadata["detections"].append(
                            [class_name, confidence, x1, y1, x2, y2]
                        )
                    save_image_path = os.path.join(output_dir+"/crops/", f"{class_name}/{track_id}")
                    if track_id not in d:
                        d[track_id] = 1
                    else:
                        d[track_id] += 1
                    os.makedirs(save_image_path, exist_ok=True)
                    cropped_image = frame[y1:y2, x1:x2]
                    cv2.imwrite(os.path.join(save_image_path, f"{track_id}_{d[track_id]}.jpg"), cropped_image)


                print(f"frame: {current_frame_id + 1}/{total_frame}")

                if self.temporal:
                    video_writer.write(frame)

                    # # Optional live preview.
                    # try:
                    #     detection_placeholder.image(
                    #         cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    #         channels="RGB",
                    #         use_container_width=True,
                    #     )
                    # except Exception as e:
                    #     logger.debug("Could not display frame: %s", e)

                else:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)

                    self.redis.set_value(frame_uid, image_encoding_using_pillow(frame_pil))
                    self.redis.set_value(frame_uid + "_data", json.dumps(frame_metadata))
                    data.append(frame_metadata)

        finally:
            video.release()

            if video_writer is not None:
                video_writer.release()

            detection_placeholder.empty()

        yolo = None

        if self.temporal:
            logger.info("Tracked video completed: %s", output_video_path)
            return output_video_path

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

                    for det in detections:
                        cr_id = str(uuid.uuid4())
                        class_name, conf = det[0], det[1]
                        crop_image = image.crop(det[2:])

                        caption = qwen_model.caption(crop_image, prompt)
                        image_embeddings = clip_model.img_embeddings(crop_image)
                        text_embeddings = clip_model.text_embeddings(caption)

                        crop_image_metadata = {
                            "id": cr_id,
                            "label": class_name,
                            "caption": caption,
                            "confidence": conf,
                            "parent_id": image_uid,
                            "image_embeddings": image_embeddings,
                            "text_embeddings": text_embeddings,
                        }

                        self.lanceDB.insert([crop_image_metadata])

                    print(f"Processed frame for captioning: {i + 1}/{len(data)}")

                except Exception as e:
                    print(e)
                    print(dt)
        return clip_model

    def run_clip_inference(self):
        pass

    def export_to_trt(self, model_path: str) -> str:
        """
        Return the model path to use for YOLO inference.

        If TensorRT is enabled:
            1. Reuse existing .engine if available.
            2. Otherwise export the .pt model to TensorRT.
        """

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found: {model_path}")

        if not self.use_tensorrt:
            return model_path

        engine_path = os.path.splitext(model_path)[0] + ".engine"

        if os.path.exists(engine_path):
            logger.info("Using existing TensorRT engine: %s", engine_path)
            return engine_path

        logger.info("TensorRT engine not found. Exporting %s", model_path)

        st.info("Converting the YOLO model to TensorRT. This only needs to be done once.")

        model = YOLO(model_path)
        exported_path = str(model.export(format="engine", device=0))

        logger.info("TensorRT model exported to: %s", exported_path)
        st.success(f"TensorRT model exported to: {exported_path}")

        return exported_path

    @staticmethod
    def _format_timestamp(milliseconds: float) -> str:
        total_seconds = int(milliseconds / 1000)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _get_caption(self, prompt_text: str, crop: Image.Image, track_id: Optional[int]) -> str:
        caption = self.qwen_model.caption(crop, prompt_text)

        if caption and track_id is not None:
            if track_id not in self._caption_cache:
                self._caption_cache[track_id] = [caption]

            if caption not in self._caption_cache[track_id]:
                self._caption_cache[track_id].append(caption)

            return self._caption_cache[track_id][-1]

        if self.caption_once_per_track and track_id is not None:
            self._caption_cache[track_id] = caption

        return caption

    def read_video_get_detections(self, video_path: str, ui, prompt_text, output_video_path: Optional[str] = None):
        """
        Run normal detection/captioning and optionally generate
        a tracked/annotated video.
        """
        self._caption_cache.clear()

        if not self.temporal:
            data = self.run_yolo_inference(video_path=video_path)
            st.success("Detections Completed")

        if not self.temporal:
            with st.spinner("Running the Captioning"):
                self.clip_model = self.run_qwen_inference(prompt_text, data)
            st.success("Clip Model Completed")

        if output_video_path and self.temporal:
            st.info("Generating tracked video...")

            tracked_video_path = self.run_yolo_inference(
                video_path=video_path,
                output_video_path=output_video_path,
            )

            st.success(f"Tracked video saved to: {tracked_video_path}")

            with st.spinner("Running the Captioning..."):
                qwen_video_reasoning = Qwen_Model(Config.QWEN_MODEL, Config.MAX_NEW_TOKENS, True)
                video_summary = qwen_video_reasoning.caption_video(tracked_video_path)
            st.success("Captioning Completed")
            return tracked_video_path, video_summary
        return data

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
