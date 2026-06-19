from ultralytics import YOLO
import cv2
import numpy as np

def calculate_scoliosis_degree():
    # --- 1. INITIAL CONFIGURATION ---
    model_path = r'[YOUR_MODEL_PATH]\best.pt'  
    model = YOLO(model_path)

    test_image_path = r'[YOUR_TEST_IMAGE_PATH]\test_image.jpg'  # Update this path to your test image

    # Run object detection inference (save=False because we will draw custom annotations)
    results = model.predict(source=test_image_path, save=False, conf=0.25)
    
    # Load original image via OpenCV for post-process annotation
    img = cv2.imread(test_image_path)
    
    # List to store the calculated center coordinates of detected vertebrae
    centers = []
    
    # --- 2. PROCESSING MODEL DETECTIONS ---
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract box coordinates (xmin, ymin, xmax, ymax)
            xyxy = box.xyxy[0].tolist()
            x_center = (xyxy[0] + xyxy[2]) / 2
            y_center = (xyxy[1] + xyxy[3]) / 2
            centers.append((x_center, y_center))
            
            # Annotate the detected vertebrae centers with small red dots
            cv2.circle(img, (int(x_center), int(y_center)), 5, (0, 0, 255), -1)

    # Sort centers vertically from top to bottom (based on the Y-axis)
    centers.sort(key=lambda point: point[1])

    # --- 3. COBB ANGLE GEOMETRIC CALCULATION ---
    if len(centers) >= 3:
        # Extract the highest, middle, and lowest vertebrae points to structure vectors
        p_top = centers[0]
        p_mid = centers[len(centers) // 2]
        p_bot = centers[-1]
        
        # Calculate standard spinal trajectory vectors
        v1 = np.array([p_mid[0] - p_top[0], p_mid[1] - p_top[1]])
        v2 = np.array([p_bot[0] - p_mid[0], p_bot[1] - p_mid[1]])
        
        # Calculate the Cobb Angle using dot product cosine similarity
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        cobb_angle = np.degrees(angle)
        
        # Overlay the final calculated degree metric onto the output image
        text = f"Estimated Cobb Angle: {cobb_angle:.1f} degrees"
        cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        print(f"\nCalculated Scoliosis Severity: {cobb_angle:.2f}°")
    else:
        print("Warning: Insufficient vertebrae points detected to calculate spinal curvature.")

    # --- 4. EXPORT RESULTS ---
    output_path = r'[YOUR_OUTPUT_PATH]\scoliosis_result.jpg'  # Update this path to your desired output location
    cv2.imwrite(output_path, img)
    print(f"Resulting analysis saved successfully to: {output_path}")

if __name__ == "__main__":
    calculate_scoliosis_degree()