import numpy as np
from PIL import Image, ImageDraw
import cv2

def highlight_text(original_img_path,data_path, preprocessed_image_path,search_text, output_img_path):
    # Load the original colorful image
    original_image = cv2.imread(original_img_path)
    orig_height, orig_width = original_image.shape[:2]

    preprocessed_image = Image.open(preprocessed_image_path)
    preprocessed_image = np.array(preprocessed_image)

    preprocessed_height, preprocessed_width = preprocessed_image.shape[:2]
    # Load Tesseract data from a text file
    with open(data_path, 'r', encoding='utf-8') as file:
        data_lines = file.readlines()

        
        # Extract data into a dictionary
        data = {
            'left': [],
            'top': [],
            'width': [],
            'height': [],
            'text': []
        }

        for line in data_lines[1:]:
            parts = line.strip().split('\t')
            if len(parts) < 12:
                continue
            data['left'].append(int(float(parts[6])* (orig_width / preprocessed_width)))
            data['top'].append(int(float(parts[7])* (orig_width / preprocessed_width)))
            data['width'].append(int(float(parts[8])* (orig_width / preprocessed_width)))
            data['height'].append(int(float(parts[9])* (orig_width / preprocessed_width)))
            data['text'].append(parts[11])

        # Flag to check if text is found
        text_found = False

        # Iterate through the detected text
        for i in range(len(data['text'])):
            text = data['text'][i]
            if search_text in text:
                text_found = True

                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])

                # Debugging: Print the coordinates and text
                print(f"Found text '{text}' at ({x}, {y}, {w}, {h})")

                # Draw a rectangle around the text on the original image
                cv2.rectangle(original_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Green rectangle

    # Save the highlighted image
    cv2.imwrite(output_img_path, original_image)

    # Print if text was found or not
    if text_found:
        print(f"Text '{search_text}' found and highlighted.")
    else:
        print(f"Text '{search_text}' not found.")

# # Example usage
# highlight_text(
#     r'D:\Document-Management-of-Nepali-Papers\results\documents\Press_Release\original_20240924132754\original_20240924132754.jpg', 
#     r'D:\Document-Management-of-Nepali-Papers\results\documents\Press_Release\original_20240924132754\original_20240924132754_preprocessed.png',
#     'मिति', 
#     r'D:\Document-Management-of-Nepali-Papers\results\documents\Press_Release\original_20240924132754\temp.png'
# )
