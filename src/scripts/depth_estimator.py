import torch
import numpy as np
import cv2
from collections import defaultdict

from PIL import Image
from depth_pro import create_model_and_transforms

class MLDepthProEstimator:
  def __init__(self, device = None, enable_optimisation = True):
    self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    self.model = None
    self.transform = None
    self.optimisation = enable_optimisation
    self.load_model()
    
    # Tracking states
    self.tracked_objects = {}
    self.prev_distances = defaultdict(lambda: None)
    self.global_object_counter = 0
    self.current_frame_num = 0
    
    # Optimisation settings
    self.frame_skip = 5
    self.last_depth_map = None
    self.depth_frame_counter = 0

  # Load depth-pro model
  def load_model(self):
    try:
      self.model, self.transform = create_model_and_transforms()
      self.model.to(self.device)
      self.model.eval()
      
      if self.optimisation:
        try:
          # Enable half precision
          if self.device.type == "mps":
            self.model = self.model.half()
            print("Half precision enabled for faster inference")
            
          if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
            print("Torch compilation enabled")
            
        except Exception as e:
          print(f"Optimisation warning: {e}")
      
      print("ML-Depth-Pro model loaded successfully")
    except ImportError:
      print("ML-Depth-Pro not installed")
      raise
    except Exception as e:
      print(f"Error loading ML-Depth Pro: {e}")
      raise
    
  def estimate_depth(self, rgb_image, force_update = False):
    self.depth_frame_counter += 1
    
    if not force_update and self.optimisation:
      if self.depth_frame_counter % self.frame_skip != 0 and self.last_depth_map is not None:
        return self.last_depth_map, None
    
    try:
      if isinstance(rgb_image, np.ndarray):
        h, w = rgb_image.shape[:2]
        if max(h,w) > 1024:
          scale = 1024 / max(h,w)
          new_h, new_w = int(h * scale), int(w * scale)
          rgb_image = cv2.resize(rgb_image, (new_w, new_h))
        
        pil_image = Image.fromarray(rgb_image)
      else:
        pil_image = rgb_image
            
      # Apply transforms
      image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
      if self.optimisation and self.device.type == 'mps':
        image_tensor = image_tensor.half()
            
      with torch.no_grad():
        # Depth Pro returns metric depth directly
        prediction = self.model.infer(image_tensor)
                
        depth_map = prediction['depth'].squeeze().cpu().numpy()
        
        if isinstance(rgb_image, np.ndarray):
          original_h, original_w = rgb_image.shape[:2]
          if depth_map.shape != (original_h, original_w):
            depth_map = cv2.resize(depth_map, (original_w, original_h))
        
        focal_length = prediction.get('focal_length', None)
        
        # Cache the results
        self.last_depth_map = depth_map
                
      return depth_map, focal_length
            
    except Exception as e:
      print(f"Depth estimation failed: {e}")
      return None, None
    
  def extract_object_distance(self, depth_map, bbox, method):    
    x1, y1, x2, y2 = bbox
    
    # Ensure valid box
    h, w = depth_map.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
      return None
    
    # Extract depth patch
    depth_patch = depth_map[y1:y2, x1:x2]
    
    # Filter out invalid depths (0 or negative)
    valid_depths = depth_patch[depth_patch > 0.1]
    
    if len(valid_depths) == 0:
      return None
    
    if method == 'median':
      return float(np.median(valid_depths))
    elif method == 'mean':
      return float(np.mean(valid_depths))
    elif method == 'robust_mean':
      # Remove outliers and take mean
      q25, q75 = np.percentile(valid_depths, [25, 75])
      iqr = q75 - q25
      lower_bound = q25 - 1.5 * iqr
      upper_bound = q75 + 1.5 * iqr
      filtered = valid_depths[(valid_depths >= lower_bound) & (valid_depths <= upper_bound)]
      return float(np.mean(filtered)) if len(filtered) > 0 else float(np.median(valid_depths))
    elif method == 'center_point':
      # Use depth at center of bounding box
      center_y, center_x = (y1 + y2) // 2, (x1 + x2) // 2
      return float(depth_map[center_y, center_x]) if depth_map[center_y, center_x] > 0.1 else None
    else:
      return float(np.median(valid_depths))
    
  # Process detections and assign distances
  def process_detections(self, detections, depth_map, camera_matrix = None):
    self.current_frame_num += 1
    
    for det in detections:
      bbox = det['bbox']
      label = det['label']
      
      # Extract distances from depth-pro
      distance_ml = self.extract_object_distance(depth_map, bbox, method = 'robust_mean')
      
      # Assign object ID
      obj_id = self.assign_object_id(label, bbox)
      
      # Smooth distance over time
      distance_final = distance_ml
      if distance_final is not None:
        prev_dist = self.prev_distances.get(obj_id)
        if prev_dist is not None:
          # Exponential moving average (EMA)
          alpha = 0.3
          distance_final = alpha * distance_final + (1 - alpha) * prev_dist
          
        self.prev_distances[obj_id] = distance_final
        
      # Update detection
      det.update({
        'object_id': obj_id,
        'distance_m': round(distance_final, 2) if distance_final else None,
        'risk': self.classify_risk(distance_final)
      })
      
    return detections
  
  # Calculate Intersection over Union (IoU) of two bounding boxes
  def iou(self, a, b, epsilon = 1e-6):
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    inter_w = max(0, min(xa2, xb2) - max(xa1, xb1))
    inter_h = max(0, min(ya2, yb2) - max(ya1, yb1))
    inter = inter_w * inter_h
    union = float((xa2 - xa1)*(ya2 - ya1) + (xb2 - xb1)*(yb2 - yb1) - inter)
    
    return inter / (union + epsilon)
  
  def assign_object_id(self, label, bbox):    
    best_iou, best_id = 0.0, None
    for obj_id, tracked in self.tracked_objects.items():
      if tracked['label'] != label:
        continue
      iou_score = self.iou(bbox, tracked['bbox'])
      if iou_score > best_iou:
        best_iou = iou_score
        best_id = obj_id
    
    if best_iou >= 0.4:
      # Update existing tracked object's info
      self.tracked_objects[best_id]['bbox'] = bbox
      self.tracked_objects[best_id]['last_seen'] = self.current_frame_num
      return best_id
    else:
      # Assign new global unique ID because no match exists
      self.global_object_counter += 1
      self.tracked_objects[self.global_object_counter] = {
        'label': label, 'bbox': bbox, 'last_seen': self.current_frame_num
      }
      return self.global_object_counter 
    
  # Classify risk based on distance
  def classify_risk(self, distance):
    if distance is None:
      return "Unknown"
    elif distance < 5.0:
      return "Dangerous"
    elif distance < 15.0:
      return "Cautious"
    else:
      return "Safe"