import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time

class SnapshotFruitCounter:
    def __init__(self, model_path, min_confidence=0.5, snapshot_interval=2.0, min_distance=80):
        """
        Snapshot-based fruit counter - counts fruits at specific moments
        
        Args:
            model_path: Path to YOLO model
            min_confidence: Minimum confidence for detections
            snapshot_interval: Seconds between automatic snapshots
            min_distance: Minimum distance between fruits to be considered separate
        """
        self.yolo = YOLO(model_path)
        self.min_confidence = min_confidence
        self.snapshot_interval = snapshot_interval
        self.min_distance = min_distance
        
        self.last_snapshot_time = 0
        self.snapshot_counts = defaultdict(int)
        self.current_detections = []
        
    def get_box_center(self, box):
        """Get center point of bounding box"""
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def is_in_region(self, center, region_points):
        """Check if point is inside the defined region"""
        x, y = center
        n = len(region_points)
        inside = False
        
        p1x, p1y = region_points[0]
        for i in range(1, n + 1):
            p2x, p2y = region_points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def remove_overlapping_detections(self, detections):
        """Remove detections that are too close to each other (same fruit detected multiple times)"""
        if len(detections) <= 1:
            return detections
        
        # Group detections by class
        by_class = defaultdict(list)
        for det in detections:
            by_class[det['class']].append(det)
        
        final_detections = []
        
        # For each class, remove overlapping detections
        for class_name, class_detections in by_class.items():
            if len(class_detections) == 1:
                final_detections.extend(class_detections)
                continue
            
            # Sort by confidence (highest first)
            class_detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            kept_detections = []
            for current_det in class_detections:
                too_close = False
                for kept_det in kept_detections:
                    distance = self.calculate_distance(current_det['center'], kept_det['center'])
                    if distance < self.min_distance:
                        too_close = True
                        break
                
                if not too_close:
                    kept_detections.append(current_det)
            
            final_detections.extend(kept_detections)
        
        return final_detections
    
    def update(self, detections, region_points):
        """Update with new detections"""
        # Get valid detections in region
        valid_detections = []
        for detection in detections:
            if detection.conf < self.min_confidence:
                continue
                
            box = detection.xyxy[0].cpu().numpy()
            center = self.get_box_center(box)
            class_name = self.yolo.names[int(detection.cls)]
            
            if self.is_in_region(center, region_points):
                valid_detections.append({
                    'center': center,
                    'class': class_name,
                    'box': box,
                    'confidence': float(detection.conf)
                })
        
        # Remove overlapping detections (same fruit detected multiple times)
        self.current_detections = self.remove_overlapping_detections(valid_detections)
        
        # Check if it's time for automatic snapshot
        current_time = time.time()
        if current_time - self.last_snapshot_time >= self.snapshot_interval:
            self.take_snapshot()
            self.last_snapshot_time = current_time
    
    def take_snapshot(self):
        """Take a snapshot of current counts"""
        counts = defaultdict(int)
        for detection in self.current_detections:
            counts[detection['class']] += 1
        
        # Update snapshot counts (take the maximum seen)
        for fruit_type, count in counts.items():
            if count > self.snapshot_counts[fruit_type]:
                self.snapshot_counts[fruit_type] = count
    
    def manual_snapshot(self):
        """Manually take a snapshot"""
        self.take_snapshot()
        print("Manual snapshot taken!")
        current_counts = defaultdict(int)
        for detection in self.current_detections:
            current_counts[detection['class']] += 1
        
        print("Current visible fruits:")
        total = 0
        for fruit_type, count in current_counts.items():
            print(f"  {fruit_type}: {count}")
            total += count
        print(f"  Total: {total}")
    
    def get_current_counts(self):
        """Get current frame counts"""
        counts = defaultdict(int)
        for detection in self.current_detections:
            counts[detection['class']] += 1
        return dict(counts)
    
    def draw_results(self, frame, region_points):
        """Draw results on frame"""
        # Draw region
        pts = np.array(region_points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        
        # Draw current detections
        for i, detection in enumerate(self.current_detections):
            box = detection['box']
            x1, y1, x2, y2 = map(int, box)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw label
            label = f"{detection['class']} #{i+1} ({detection['confidence']:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw current counts
        y_offset = 30
        cv2.putText(frame, "CURRENT FRAME:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        current_counts = self.get_current_counts()
        current_total = 0
        for fruit_type, count in current_counts.items():
            cv2.putText(frame, f"{fruit_type}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
            current_total += count
        
        cv2.putText(frame, f"Current Total: {current_total}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        y_offset += 40
        
        # Draw snapshot counts (max ever seen)
        cv2.putText(frame, "SNAPSHOT COUNTS (MAX):", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += 30
        
        snapshot_total = 0
        for fruit_type, count in self.snapshot_counts.items():
            cv2.putText(frame, f"{fruit_type}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25
            snapshot_total += count
        
        cv2.putText(frame, f"MAX TOTAL: {snapshot_total}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
        
        # Show time until next snapshot
        current_time = time.time()
        time_until_snapshot = self.snapshot_interval - (current_time - self.last_snapshot_time)
        if time_until_snapshot > 0:
            cv2.putText(frame, f"Next snapshot in: {time_until_snapshot:.1f}s", (10, frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def reset_counts(self):
        """Reset all counts"""
        self.snapshot_counts.clear()
        self.current_detections.clear()
        self.last_snapshot_time = 0

def count_unique_fruits(camera_id=2, model_path='fruitDetect.pt'):
    """Count unique fruits using snapshot approach"""
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_id)
    assert cap.isOpened(), f"Error: Could not open camera {camera_id}"
    
    # Get camera properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Camera resolution: {w}x{h} at {fps} FPS")
    
    # Define counting region
    region_points = [(0, 0), (0, h-25), (w-5, h-25), (w-5, 0)]
    
    # Initialize counter
    counter = SnapshotFruitCounter(
        model_path=model_path,
        min_confidence=0.5,     # Higher confidence for more reliable detections
        snapshot_interval=3.0,  # Take snapshot every 3 seconds
        min_distance=100        # Minimum distance between fruits (adjust based on fruit size)
    )
    
    print("Starting snapshot-based fruit counting...")
    print("Controls:")
    print("  'q' - Quit")
    print("  'r' - Reset all counts")
    print("  'SPACE' - Take manual snapshot")
    print("  '+' - Increase min distance between fruits")
    print("  '-' - Decrease min distance between fruits")
    print("\nHow it works:")
    print("- Automatically takes snapshots every 3 seconds")
    print("- 'Current Total' shows fruits visible right now")
    print("- 'MAX TOTAL' shows the highest count ever detected")
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to read frame")
                break
            
            # Run YOLO detection
            results = counter.yolo(frame, verbose=False)
            
            # Update counter
            if len(results) > 0 and len(results[0].boxes) > 0:
                counter.update(results[0].boxes, region_points)
            else:
                counter.update([], region_points)
            
            # Draw results
            annotated_frame = counter.draw_results(frame, region_points)
            
            # Display frame
            cv2.imshow("Snapshot Fruit Counter", annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                counter.reset_counts()
                print("All counts reset!")
            elif key == ord(' '):  # Spacebar
                counter.manual_snapshot()
            elif key == ord('+') or key == ord('='):
                counter.min_distance += 20
                print(f"Min distance increased to: {counter.min_distance}")
            elif key == ord('-'):
                counter.min_distance = max(40, counter.min_distance - 20)
                print(f"Min distance decreased to: {counter.min_distance}")
            elif key == ord('s'):
                filename = f"snapshot_{int(time.time())}.png"
                cv2.imwrite(filename, annotated_frame)
                print(f"Image saved as {filename}")

    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        # Print final results
        print("\n" + "="*50)
        print("FINAL RESULTS:")
        print("="*50)
        
        # Take final snapshot
        counter.take_snapshot()
        
        print("Maximum fruits detected:")
        total_count = 0
        for fruit_type, count in counter.snapshot_counts.items():
            print(f"  {fruit_type}: {count}")
            total_count += count
        
        print(f"\nTotal unique fruits: {total_count}")
        
        print("\nCurrent visible fruits:")
        current_counts = counter.get_current_counts()
        current_total = 0
        for fruit_type, count in current_counts.items():
            print(f"  {fruit_type}: {count}")
            current_total += count
        print(f"Current total: {current_total}")
        
        print("="*50)
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

# Usage
if __name__ == "__main__":
    count_unique_fruits(camera_id=2, model_path='fruitDetect.pt')