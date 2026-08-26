from ultralytics import YOLO
import cv2
import argparse
import os

def detect_image(model, image_path, conf_threshold=0.05):
    """Detect objects in an image"""
    results = model.predict(image_path, conf=conf_threshold)
    return results[0]

def detect_webcam(model, conf_threshold=0.05):
    """Detect objects using webcam"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        # Perform detection with lower confidence threshold
        results = model.predict(frame, conf=conf_threshold)
        
        # Get the annotated frame
        annotated_frame = results[0].plot()
        
        # Display the frame
        cv2.imshow("YOLOv8 Detection", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Object Detection')
    parser.add_argument('--source', type=str, default='webcam',
                      help='Source for detection: "webcam" or path to image/video')
    parser.add_argument('--conf', type=float, default=0.05,
                      help='Confidence threshold (default: 0.05)')
    args = parser.parse_args()
    
    # Load the model
    model = YOLO('runs/detect/train/weights/best.pt')
    
    if args.source.lower() == 'webcam':
        print("Starting webcam detection... Press 'q' to quit")
        detect_webcam(model, args.conf)
    else:
        if not os.path.exists(args.source):
            print(f"Error: Source file {args.source} does not exist")
            return
            
        print(f"Processing {args.source}...")
        results = detect_image(model, args.source, args.conf)
        
        # Save the results
        output_dir = 'runs/detect/predict'
        os.makedirs(output_dir, exist_ok=True)
        results.save(output_dir)
        print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main() 