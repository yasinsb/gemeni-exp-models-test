import json
import cv2

PROMPT = """
Given the image you will receive, return bounding boxes as a JSON array, grouping peppers by their name and level of hotness
levels of hotness annotated with color:
  - green: not too hot
  - yellow: somewhat hot
  - yellow: pretty hot
  - red: very hort
"""

# Constants for display settings
MAX_WINDOW_WIDTH = 1512
MAX_WINDOW_HEIGHT = 982
FONT_SCALE_FACTOR = 900  # Increased to prevent excessive scaling
FONT_THICKNESS = 2  # Slightly thicker for clarity
BOX_THICKNESS = 2
TEXT_PADDING_X = 5  # Reduced to prevent background overflow
TEXT_PADDING_Y = 4  # Adjusted for better alignment

COLOR_MAPPING = {
    "green": (76, 177, 34),    # BGR for bright green
    "yellow": (0, 215, 255),   # BGR for strong golden yellow
    "orange": (0, 140, 255),   # BGR for deep orange
    "red": (60, 20, 220)       # BGR for vivid crimson red
}

def get_label_color(color):
    """Returns a fixed color for a given label color."""
    return COLOR_MAPPING.get(color.lower(), (0, 0, 0))  # Default to black

def validate_json(json_data):
    """Validates the JSON data structure for multiple bounding boxes."""
    for item in json_data:
        if "title" not in item or "coords" not in item:
            print(f"Error: Missing 'title' or 'coords' in {item}")
            return False
        if not isinstance(item["title"], str):
            
            print(f"Error: 'title' must be a string, found {type(item['title'])}")
            return False
        if not isinstance(item["coords"], list):
            print(f"Error: 'coords' must be a list, found {type(item['coords'])}")
            return False

        # Check each bounding box in coords
        for box in item["coords"]:
            if not (isinstance(box, (list, tuple)) and len(box) == 4):
                print(f"Error: Each box must be [y1, x1, y2, x2], found {box}")
                return False
            if not all(isinstance(val, (int, float)) for val in box):
                print(f"Error: Box values must be numbers, found {box}")
                return False

    return True

def draw_boxes(image, json_data, show_annotations=True):
    """Draw bounding boxes (multiple per item) with text labels."""
    img_h, img_w = image.shape[:2]

    # Resize image to fit within window if needed
    scale_factor = min(MAX_WINDOW_WIDTH / img_w, MAX_WINDOW_HEIGHT / img_h)
    if scale_factor < 1:
        new_w = int(img_w * scale_factor)
        new_h = int(img_h * scale_factor)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img_h, img_w = new_h, new_w

    annotated_img = image.copy()

    for item in json_data:
        title = item["title"]
        color = item.get("color", "black")
        color = get_label_color(color)

        # Draw each bounding box
        for (y1, x1, y2, x2) in item["coords"]:
            # Convert normalized (0-1000) to absolute pixels
            Y1, Y2 = int(y1 / 1000 * img_h), int(y2 / 1000 * img_h)
            X1, X2 = int(x1 / 1000 * img_w), int(x2 / 1000 * img_w)

            cv2.rectangle(annotated_img, (X1, Y1), (X2, Y2), color, BOX_THICKNESS, cv2.LINE_AA)

            if show_annotations:
                # Auto-scale font
                # Auto-scale font
                font_scale = max(0.6, min(img_w, img_h) / FONT_SCALE_FACTOR)  # Reduced max scale
                thickness = max(1, int(font_scale * FONT_THICKNESS))  # Adjusted thickness

                # Measure text
                text_size, _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                text_w, text_h = text_size

                # Ensure background fits text without excessive space
                cv2.rectangle(
                    annotated_img,
                    (X1, Y1 - text_h - TEXT_PADDING_Y * 2),
                    (X1 + text_w + TEXT_PADDING_X * 2, Y1),
                    color,
                    -1
                )

                # Choose text color
                brightness = sum(color) / 3
                text_color = (0, 0, 0) if brightness > 150 else (255, 255, 255)

                # Draw label text
                cv2.putText(
                    annotated_img,
                    title,
                    (X1 + TEXT_PADDING_X, Y1 - TEXT_PADDING_Y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    text_color,
                    thickness,
                    cv2.LINE_AA
                )

    return annotated_img

def main():
    image_path = "peppers3.png"  # Your image
    json_file = "data.json"      # Your JSON

    # Load data
    with open(json_file, "r") as f:
        data = json.load(f)
    if not validate_json(data):
        return

    # Load image
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Error: Could not read image from {image_path}")
        return

    while True:
        annotated_img = draw_boxes(original_img, data, show_annotations=True)
        cv2.imshow("Peppers", annotated_img)

        # Wait for key press
        key = cv2.waitKey(0)
        if key != -1:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()