import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np

# --- 1. Model Uploading ---
# Uploading the model outside so that it doesn't have to be reloaded for each image.
MODEL_PATH = r'[YOUR_MODEL_PATH]\best.pt'  # Update this path to your trained model
model = YOLO(MODEL_PATH)

def process_xray(img):
    #Converts the gradio images to RGB Numpy format.
    
    # YOLO Prediction (Giving the image directly to the model)
    results = model.predict(source=img, conf=0.25)
    
    centers = []
    output_img = img.copy() # Draw on a copy of the original image to avoid modifying it directly.
    
    # --- 2. VERTEBRA DETECTION AND DRAWING ---
    for r in results:
        boxes = r.boxes
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            x_center = (xyxy[0] + xyxy[2]) / 2
            y_center = (xyxy[1] + xyxy[3]) / 2
            centers.append((x_center, y_center))
            
            # Since Gradio uses RGB, we use (255, 0, 0) for red. (OpenCV is also a BGR)
             
            cv2.circle(output_img, (int(x_center), int(y_center)), 5, (255, 0, 0), -1)

    centers.sort(key=lambda point: point[1])
    
    # --- 3. COBB ANGLE CALCULATION ---
    result_text = ""
    if len(centers) >= 3:
        p_top = centers[0]
        p_mid = centers[len(centers) // 2]
        p_bot = centers[-1]
        
        v1 = np.array([p_mid[0] - p_top[0], p_mid[1] - p_top[1]])
        v2 = np.array([p_bot[0] - p_mid[0], p_bot[1] - p_mid[1]])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        cobb_angle = np.degrees(angle)
        
        # Overwrite the degree onto image (Green)
        text = f"Cobb Angle: {cobb_angle:.1f} degrees"
        cv2.putText(output_img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Prepare a report for text within interface
        result_text = f"Analysis is successful!\nPredicted Scoliosis Angle: {cobb_angle:.2f}°\nDetected Vertebrae Count: {len(centers)}"
    else:
        result_text = "Warning: Not enough vertebrae detected to calculate the angle."

    return output_img, result_text

# --- 4. WEB INTERFACE DESIGN ---
interface = gr.Interface(
    fn=process_xray,
    inputs=gr.Image(type="numpy", label="1. Upload X-ray Image"),
    outputs=[
        gr.Image(type="numpy", label="2. Processed Result"),
        gr.Textbox(label="3. System Report")
    ],
    title="Scoliosis Analysis & Cobb Angle Predictor",
    description="This application uses deep learning (YOLO) to detect vertebrae in X-ray images and calculates the Cobb angle between them.",
    theme="default"
)

if __name__ == "__main__":
    # share=True gives you a temporary link accessible from the internet.
    interface.launch(share=False)