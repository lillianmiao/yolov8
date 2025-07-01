import cv2
from ultralytics import YOLO
from ultralytics import solutions

cap = cv2.VideoCapture(2)

yolo = YOLO('fruitDetect.pt')  # Load your trained model

def count_objects_in_region(video_path, output_video_path, model_path):
    """Count objects in a specific region within a video."""
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    #video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    region_points = [(0, 0), (0, 475), (635, 475), (635, 0)]
    counter = solutions.ObjectCounter(show=True, region=region_points, model=model_path)

    while cap.isOpened():
        success, im0 = cap.read()
        results = counter(im0)

        

        #video_writer.write(results.plot_im)
        #cv2.imshow("Object Counter", results.plot_im)
        if cv2.waitKey(150) & 0xFF == ord('q'):
            print(f"Unique objects detected: {results.count}")
            break

    cap.release()
    cv2.destroyAllWindows()


count_objects_in_region("path/to/video.mp4", "output_video.avi", "fruitDetect.pt")
