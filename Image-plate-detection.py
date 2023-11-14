import cv2
import pytesseract
from PIL import Image
import csv
from yolov8 import YOLOv8

# Set Tesseract OCR executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Faizan Nadeem\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Function to initialize YOLOv8 detector
def initialize_yolov8_detector(model_path, conf_thres=0.25, iou_thres=0.3):
    return YOLOv8(model_path, conf_thres=conf_thres, iou_thres=iou_thres)

# Function to load an image from a file path
def load_image_from_path(path):
    return cv2.imread(path)

# Function to detect objects in an image
def detect_objects(yolov8_detector, img):
    return yolov8_detector(img)

# Function to draw detected objects on an image
def draw_detections(yolov8_detector, img, boxes):
    return yolov8_detector.draw_detections(img, boxes)

# Function to extract a region from an image based on coordinates
def extract_region(image, coordinates):
    x_min, y_min, x_max, y_max = [int(coord) for coord in coordinates]
    return image[y_min:y_max, x_min:x_max]

# Function to read text from an image using Tesseract OCR
def read_text_from_image(image):
    return pytesseract.image_to_string(image, lang='eng', config='--psm 6')

# Function to save text to a CSV file while ensuring uniqueness
def save_text_to_csv(text, csv_file, unique_text_set):
    if text and text not in unique_text_set:
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            index = get_last_index(csv_file) + 1
            writer.writerow([index, text])
            unique_text_set.add(text)
            print(f"Data saved with index {index}")
    else:
        print("No text found or duplicate text skipped")

# Function to get the last index in the CSV file
def get_last_index(csv_file):
    try:
        with open(csv_file, 'r') as file:
            lines = list(csv.reader(file))
            if lines:
                last_row = lines[-1]
                return int(last_row[0])
    except FileNotFoundError:
        return 0
    return 0

# Main function for processing the image, detecting objects, and extracting text
def main(image_path, model_path, csv_file):
    # Initialize YOLOv8 detector
    yolov8_detector = initialize_yolov8_detector(model_path)
    
    # Load the image from the given file path
    img = load_image_from_path(image_path)
    
    # Detect objects in the image
    boxes, _, _ = detect_objects(yolov8_detector, img)
    
    # Draw detected objects on the image
    combined_img = draw_detections(yolov8_detector, img, boxes)

    # Create a window to display the image with detected objects
    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
    cv2.imshow("Detected Objects", combined_img)

    # Extract and process text from the detected region if objects are found
    if len(boxes) > 0:
        coordinates = boxes[0]
        roi = extract_region(img, coordinates)
        cv2.imshow("Extracted Region", roi)
        cv2.waitKey(0)

        # Extract text from the region and save it to the CSV file
        extracted_text = read_text_from_image(roi)
        save_text_to_csv(extracted_text, csv_file, unique_text_set)

    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Specify the file paths and initialize a set for unique text values
    image_path = 'np2.jpg'
    model_path = "models/best.onnx"
    csv_file = 'data.csv'
    unique_text_set = set()

    # Call the main function to process the image
    main(image_path, model_path, csv_file)
