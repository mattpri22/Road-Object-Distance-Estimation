from ultralytics import YOLO
import time
import numpy as np
from .depth_estimator import MLDepthProEstimator
import cv2

# Defining COCO classes relevant for road/driving scenarios
ROAD_CLASSES = {
  0: 'person',
  1: 'bicycle', 
  2: 'car',
  3: 'motorcycle',
  5: 'bus',
  7: 'truck'
}

class DistanceEstimator:
  def __init__(self, yolo_model = "yolo11s.pt", enable_optimisation = True):
    self.yolo_model = YOLO(yolo_model)
    self.ml_depth_pro = MLDepthProEstimator(enable_optimisation = enable_optimisation)
    self.enable_optimisation = enable_optimisation
    
    # Configure for distance estimation
    self.yolo_model.overrides.update({
      'conf': 0.35, # Lower threshold for safety
      'iou': 0.45,
      'max_det': 50,
      'classes': list(ROAD_CLASSES.keys())
    })
    
    self.yolo_times = []
    self.depth_times = []
  
  # Clip bounding box to frame boundaries
  def clip_bbox(self, bbox, frame_shape):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    return (max(0,x1), max(0, y1), min(w - 1, x2), min(h - 1, y2))
  
  # Detect objects using YOLO with timing
  def detect_objects(self, frame, conf_thres = 0.35):
    start_time = time.time()
    results = self.yolo_model(frame)[0] # Get the first result (single image)
    detections = []
    
    for box in results.boxes:
      class_id = int(box.cls.item())
      if class_id in ROAD_CLASSES:
        confidence = float(box.conf.item())
        if confidence < conf_thres:
          continue
      
        # Extract bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])
      
        # Clip bounding box to frame size
        x1, y1, x2, y2 = self.clip_bbox((x1,y1,x2,y2), frame.shape)
        
        if x2 <= x1 or y2 <= y1:
          continue # Skip invalid box
      
        label = ROAD_CLASSES[class_id]
        detections.append({
          "bbox": (x1,y1,x2,y2),
          "label": label,
          "confidence": confidence
        })
      
    # Track YOLO performance
    yolo_time = time.time() - start_time
    self.yolo_times.append(yolo_time)
    if len(self.yolo_times) > 100:
      self.yolo_times.pop(0)  
      
    return detections
  
  # Process a single frame
  def process_frame(self, frame, force_depth_update = False):
    detections = self.detect_objects(frame)
    
    # Depth Pro depth estimation
    depth_start = time.time()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    depth_map, _ = self.ml_depth_pro.estimate_depth(rgb_frame, force_depth_update)
    
    depth_time = time.time() - depth_start
    self.depth_times.append(depth_time)
    if len(self.depth_times) > 100:
      self.depth_times.pop(0)
      
    if depth_map is not None:
      detections = self.ml_depth_pro.process_detections(detections, depth_map)
      
    return detections, depth_map
  
  # Get performance statistics
  def get_performance_stats(self):
    avg_yolo = np.mean(self.yolo_times) if self.yolo_times else 0
    avg_depth = np.mean(self.depth_times) if self.depth_times else 0
    return {
        'avg_yolo_time': avg_yolo,
        'avg_depth_time': avg_depth,
        'avg_total_time': avg_yolo + avg_depth,
        'theoretical_fps': 1.0 / (avg_yolo + avg_depth) if (avg_yolo + avg_depth) > 0 else 0
    }  
  
  # Draw detections on frame
  def draw_detections(self, frame, detections, depth_map = None):
    for det in detections:
      x1, y1, x2, y2 = det['bbox']
      label = det['label']
      confidence = det['confidence']
      distance = det.get('distance_m')
      risk = det.get('risk', 'Unknown')
      obj_id = det.get('object_id', 'N/A')
      
      # Color based on risk
      risk_colors = {
          'Dangerous': (0, 0, 255),    # Red
          'Cautious': (0, 165, 255),   # Orange
          'Safe': (0, 255, 0),         # Green
          'Unknown': (128, 128, 128)   # Gray
      }
      color = risk_colors.get(risk, (128, 128, 128))
            
      # Draw bounding box
      cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
      # Prepare text
      text_lines = [
          f"{label} ({confidence:.2f})",
          f"ID: {obj_id}",
          f"{distance:.1f}m" if distance else "N/A",
          f"{risk}"
      ]
      
      font = cv2.FONT_HERSHEY_SIMPLEX
      font_scale = 0.5
      font_thickness = 1

      # Draw text background and text
      y_offset = y1 - 10
      for i, text in enumerate(text_lines):
          (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
          # Draw filled rectangle for background
          cv2.rectangle( frame, (x1, y_offset - text_height - baseline), (x1 + text_width, y_offset + baseline), color, -1)
          # Draw the text on top of the rectangle
          cv2.putText(frame, text, (x1, y_offset), font, font_scale, (255, 255, 255), font_thickness, lineType = cv2.LINE_AA)
          y_offset -= (text_height + baseline + 5)  # Move up for next line
        
    return frame