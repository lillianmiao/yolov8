import cv2
from ultralytics import YOLO

# Load the model
yolo = YOLO('fruitDetect.pt') # fruit detection model trained on custom dataset in Google Colab

# Load the video capture
videoCap = cv2.VideoCapture(0) # change the index to your camera or video file path

total_fruit_count = 0  # Initialize total fruit count

# Function to get class colors
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
    (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

def detect_and_draw(frame):
    results = yolo.predict(frame, stream=False)
    fruit_count = 0
    
    for result in results:
        classes_name = result.names
        for box in result.boxes:
            if box.conf[0] > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                colour = getColours(cls)
                class_name = classes_name[cls]

                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(frame, f'{class_name} {box.conf[0]:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
                fruit_count += 1

    return frame, fruit_count
capture_index = 0

while True:
    ret, frame = videoCap.read()
    if not ret:
        continue
    results = yolo.track(frame, stream=True)

    current_count = 0

    for result in results:
        # get the classes names
        classes_names = result.names

        # iterate over each box
        for box in result.boxes:
            # check if confidence is greater than 40 percent
            if box.conf[0] > 0.4:
                # get coordinates
                [x1, y1, x2, y2] = map(int, box.xyxy[0])
                # get the class
                cls = int(box.cls[0])

                # get the class name
                class_name = classes_names[cls]

                # get the respective colour
                colour = getColours(cls)

                # draw the rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                # put the class name and confidence on the image
                cv2.putText(frame, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)

                current_count += 1

                
    # show the image
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    # break the loop if 'q' is pressed
    if key == ord('x'):
        break

    elif key == ord('c'):
        print("[INFO] Capturing frame for fruit count...")
        captured_frame = frame.copy()
        result_frame, count = detect_and_draw(captured_frame)

        total_fruit_count += count

        cv2.imwrite(f'capture_{capture_index}.jpg', result_frame)
        capture_index += 1

        cv2.imshow('Captured Detection', result_frame)
        print(f'[INFO] Detected {count} fruits in the captured frame.')

# release the video capture and destroy all windows
videoCap.release()
cv2.destroyAllWindows()
