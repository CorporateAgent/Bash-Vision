import json
from ultralytics import YOLOWorld

# Load categories from a JSON file
def load_categories(json_path):
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
        return data['category-3']

# Perform inference using the YOLO model
def perform_inference(model_path, image_path=None, categories=None):
    # If image_path is not provided, use a default
    if image_path is None:
        image_path = "data/images/05.jpeg"  # Default image path

    # If categories are not provided, load from the default JSON file
    if categories is None:
        json_path = "data/results/category-3.json"
        categories = load_categories(json_path)

    # Load the model and set the categories
    model = YOLOWorld(model_path)
    model.set_classes(categories)
    
    # Run the inference on the provided image
    results = model(image_path, 
                    conf=0.1,         # Confidence threshold
                    iou=0.5,           # IoU threshold for Non-Max Suppression
                    imgsz=640,         # Image size for inference
                    device="cpu",      # Use CPU for inference
                    save=True,         # Save annotated image
                    show=True,         # Display image with predictions
                    line_width=1,      # Line width for bounding boxes
                    save_crop=True,    # Save cropped objects
                    save_txt=True,     # Save results to a .txt file
                    save_conf=True,    # Save confidence scores in the .txt file
                    show_conf=True,    # Show confidence scores on the image
                    max_det=5,         # Maximum number of detections
                    retina_masks=True) # Use masks if the model supports them
    
    return results

# Log detection results to the console
def log_results(results):
    for result in results:
        boxes = result.boxes
        names = result.names
    
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls)  # Class ID
                class_name = names[cls_id]  # Class name
                confidence = box.conf.item()  # Confidence score
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box coordinates
                # Log the detection
                print(f"Detected {class_name} with confidence {confidence:.2f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

# Main function to execute the script directly
if __name__ == "__main__":
    model_path = "data/models/YOLO-World/YOLOv8x Worldv2.pt"
    
    # Example usage: Perform inference on a default image path
    results = perform_inference(model_path)
    log_results(results)
    results[0].show()