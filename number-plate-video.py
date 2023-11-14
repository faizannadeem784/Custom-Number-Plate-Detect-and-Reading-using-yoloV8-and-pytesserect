import cv2
import csv
from yolov8 import YOLOv8
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Faizan Nadeem\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

def initialize_video_capture(path):
    return cv2.VideoCapture(path)

def initialize_yolov8_model(model_path):
    return YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

def extract_region(frame, coordinates):
    x_min, y_min, x_max, y_max = [int(coord) for coord in coordinates]
    return frame[y_min:y_max, x_min:x_max]

def detect_objects_and_draw(frame, yolov8_detector):
    boxes, _, _ = yolov8_detector(frame)
    combined_img = yolov8_detector.draw_detections(frame)
    return boxes, combined_img

def extract_text_from_image(image):
    return pytesseract.image_to_string(image, lang='eng', config='--psm 6')

def main():
    path = 'demo.mp4'
    cap = initialize_video_capture(path)
    model_path = "models/best.onnx"
    yolov8_detector = initialize_yolov8_model(model_path)

    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
    unique_text_set = set()

    while cap.isOpened():
        if cv2.waitKey(1) == ord('q'):
            break

        try:
            ret, frame = cap.read()
            if not ret:
                break
        except Exception as e:
            print(e)
            continue

        boxes, combined_img = detect_objects_and_draw(frame, yolov8_detector)

        if len(boxes) > 0:
            coordinates = boxes[0]
            roi = extract_region(frame, coordinates)
            cv2.imshow("Extracted Region", roi)
            extracted_text = extract_text_from_image(roi)
            print(extracted_text)

            text = extracted_text
            if text:
                save_to_csv(text, unique_text_set)

        cv2.imshow("Detected Objects", combined_img)

    cap.release()
    cv2.destroyAllWindows()

def save_to_csv(text, unique_text_set):
    try:
        if text not in unique_text_set:
            with open('data.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                index = get_last_index()
                writer.writerow([index, text])
                unique_text_set.add(text)
                print(f"Data saved with index {index}")
        else:
            print("Duplicate text skipped.")
    except Exception as e:
        print(f"Error: {str(e)}")

def get_last_index():
    try:
        with open('data.csv', 'r') as file:
            lines = list(csv.reader(file))
            if lines:
                last_row = lines[-1]
                return int(last_row[0])
    except FileNotFoundError:
        return 0
    return 0

if __name__ == "__main__":
    main()
