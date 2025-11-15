from src.scripts.yolo_detect import DistanceEstimator

import numpy as np
import cv2
import os
import argparse
import time

def main():
  parser = argparse.ArgumentParser(description = 'Distance Estimation for Driving Safety')
  parser.add_argument('--video', type = str, help = 'Path to video file (optional)')
  parser.add_argument('--camera', type = int, default = 0, help = 'Camera index (default: 0)')
  parser.add_argument('--yolo_model', type = str, default = 'yolo11s.pt', help = 'YOLO model path')
  parser.add_argument('--output', type = str, help = 'Output video path (optional)')
  parser.add_argument('--show_depth', action = 'store_true', help = 'Show depth map visualization')
  parser.add_argument('--no_optimisation', action = 'store_true', help = 'Disable optimisations')
  parser.add_argument('--batch_process', action = 'store_true', help = 'Process video in batch mode (no real-time display)')
  parser.add_argument('--resize_factor', type = float, default = 1.0, help = 'Resize factor for input (0.5 = half size)')
    
  args = parser.parse_args()
  
  # Initialise distance estimator
  print("Initialising distance estimator...")
  try:
    estimator = DistanceEstimator(args.yolo_model, enable_optimisation = not args.no_optimisation)
    print("Distance estimator initialised successfully!")
  except Exception as e:
    print(f"Failed to initialise: {e}")
    return
  
  # Setup video source
  if args.video and os.path.exists(args.video):
    print(f"Using video file: {args.video}")
    cap = cv2.VideoCapture(args.video)
  else:
    print(f"Using camera {args.camera}")
    cap = cv2.VideoCapture(args.camera)
    
  if not cap.isOpened():
    print("Error: Could not open video source")
    return

  # Get video properties
  fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if args.video else float('inf')
    
  if args.resize_factor != 1.0:
    width = int(width * args.resize_factor)
    height = int(height * args.resize_factor)
    
  print(f"Video properties: {width}x{height} @ {fps}fps")
  if args.video:
    print(f"Total frames: {total_frames}")
    
  # Setup video writer if output is specified
  writer = None
  if args.output:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    print(f"Output will be saved to: {args.output}")
    
  # Main processing loop
  frame_count = 0
  processing_times = []
  start_total = time.time()
  
  if not args.batch_process:
    print("Starting processing... Press 'q' to quit ,'s' to screenshot and 'p' for performance stats")
  else:
    print("Starting batch processing")
  
  try:
    while True:
      ret, frame = cap.read()
      if not ret:
        print("End of video or camera has been disconnected")
        break
      
      # Apply resize if specified
      if args.resize_factor != 1.0:
        frame = cv2.resize(frame, (width, height))
      
      start_time = time.time()
      
      # Process frame
      detections, depth_map = estimator.process_frame(frame)
      
      # Draw detections
      annotated_frame = estimator.draw_detections(frame.copy(), detections, depth_map)
      
      # Calculate FPS
      process_time = time.time() - start_time
      processing_times.append(process_time)
      frame_count += 1
            
      # Calculate various FPS metrics
      avg_process_time = np.mean(processing_times[-30:])  # Last 30 frames
      current_fps = 1.0 / process_time if process_time > 0 else 0
      avg_fps = 1.0 / avg_process_time if avg_process_time > 0 else 0
            
      # Add performance info to display
      perf_stats = estimator.get_performance_stats()
      cv2.putText(annotated_frame, f"Current FPS: {current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
      cv2.putText(annotated_frame, f"Objects: {len(detections)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
      # Progress for video files
      if args.video and total_frames > 0:
          progress = (frame_count / total_frames) * 100
          cv2.putText(annotated_frame, f"Progress: {progress:.1f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
      # Write frame if output is specified (this maintains original fps)
      if writer:
        writer.write(annotated_frame)
            
      # Show frame in real-time mode
      if not args.batch_process:
        cv2.imshow('Distance Estimation', annotated_frame)
                
          # Show depth map if requested
        if args.show_depth and depth_map is not None:
          # Normalise depth map for visualization
          depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
          depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
          cv2.imshow('Depth Map', depth_colored)
                
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
          break
        elif key == ord('s'):
          screenshot_name = f"screenshot_{frame_count}.jpg"
          cv2.imwrite(screenshot_name, annotated_frame)
          print(f"Screenshot saved: {screenshot_name}")
        elif key == ord('p'):
          print(f"\nPerformance Stats (Frame {frame_count}):")
          print(f"  YOLO avg time: {perf_stats['avg_yolo_time']:.3f}s")
          print(f"  Depth avg time: {perf_stats['avg_depth_time']:.3f}s")
          print(f"  Total avg time: {perf_stats['avg_total_time']:.3f}s")
          print(f"  Theoretical FPS: {perf_stats['theoretical_fps']:.1f}")
            
      # Print progress every 30 frames in batch mode
      elif frame_count % 30 == 0:
        elapsed = time.time() - start_total
        eta = (elapsed / frame_count) * (total_frames - frame_count) if args.video and total_frames > 0 else 0
        print(f"Processed {frame_count} frames, Current FPS: {current_fps:.1f}, ETA: {eta:.0f}s")
            
      # Alert for dangerous objects
      dangerous_objects = [d for d in detections if d.get('risk') == 'Dangerous']
      if dangerous_objects and frame_count % 30 == 0:
        print(f"ALERT: {len(dangerous_objects)} dangerous objects detected!")
        for obj in dangerous_objects:
            print(f"  - {obj['label']} at {obj['distance_m']}m (ID: {obj['object_id']})")
              
  except KeyboardInterrupt:
    print("\nInterrupted by user")
      
  finally:
    # Cleanup
    cap.release()
    if writer:
      writer.release()
    cv2.destroyAllWindows()
        
  # Print final statistics
  total_time = time.time() - start_total
  print(f"\nProcessing complete!")
  print(f"Total frames: {frame_count}")
  print(f"Total time: {total_time:.2f}s")
  print(f"Average processing FPS: {frame_count/total_time:.2f}")
        
  if args.output:
    print(f"\nOutput video saved to: {args.output}")
    print(f"Output video plays at {fps}fps (smooth playback)")
        
  perf_stats = estimator.get_performance_stats()
  print(f"\nDetailed Performance:")
  print(f"YOLO average: {perf_stats['avg_yolo_time']:.3f}s per frame")
  print(f"Depth estimation average: {perf_stats['avg_depth_time']:.3f}s per frame")
  print(f"Total average: {perf_stats['avg_total_time']:.3f}s per frame")
  print(f"Theoretical max FPS: {perf_stats['theoretical_fps']:.1f}")
    
if __name__ == "__main__":
  main()
  
  # ---- Parameter Usage Examples -----
  # Use camera (everything default)
  # python main.py
  
  # Use video file
  # python main.py --video {path to video}
  
  # Save output and show depth map
  # python main.py --video {path to video} --output {path to output} --show_depth
  
  # Use different camera
  # python main.py --camera 1
  
  # Use custom YOLO model
  # python main.py --yolo_model {yolo11n.pt}
  
  # Example
  # python main.py --video {videos/...}.mp4 --output {annotated_videos/...}.mp4 --batch_process --resize_factor 0.5 --yolo_model yolov11s.pt
  # Video properties: 960x540 @ 24fps (If inputted video is 1080p 24 fps)
