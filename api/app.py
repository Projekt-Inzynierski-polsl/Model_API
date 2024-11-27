from flask import Flask, request, jsonify
import numpy as np
from PIL import Image, ImageOps
import cv2
import json

import seg
import ocr

app = Flask(__name__)


def convert_to_dict(coords):
    coords_dicts = []
    for coords in coords:
        coords_dicts.append(
            {
                "leftX": coords[0],
                "leftY": coords[1],
                "rightX": coords[2],
                "rightY": coords[3],
            }
        )
    return coords_dicts


def create_output(outputs, coords):
    coords_dicts = convert_to_dict(coords)

    decoded_output = []
    for i in range(len(outputs)):
        decoded_output.append(
            {
                "content": str(outputs[i][0]),
                "leftX": coords_dicts[i]["leftX"],
                "leftY": coords_dicts[i]["leftY"],
                "rightX": coords_dicts[i]["rightX"],
                "rightY": coords_dicts[i]["rightY"],
                # 'coordinates': coords_dicts[i]
            }
        )
    # return jsonify({'decoded outputs': decoded_output})
    return decoded_output


def create_final_output(outputs, coords):
    coords_dicts = convert_to_dict(coords)

    decoded_output = []
    for i in range(len(coords)):
        decoded_output.append(
            {
                # 'coordinates': coords_dicts[i],
                "leftX": coords_dicts[i]["leftX"],
                "leftY": coords_dicts[i]["leftY"],
                "rightX": coords_dicts[i]["rightX"],
                "rightY": coords_dicts[i]["rightY"],
                "lines": outputs[i],
            }
        )
    return jsonify(decoded_output)


def extract_text_regions(segmentation_mask, original_image):
    contours, _ = cv2.findContours(
        segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    text_regions = []
    coords = []
    for contour in contours:
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Crop the region from the original image
        cropped_region = original_image[y : y + h, x : x + w]
        text_regions.append(cropped_region)
        coords.append([x, y, x + w, y + h])

    return text_regions, coords


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Get the uploaded file
    image_file = request.files["image"]

    json_data = request.form.get("json")
    bounding_boxes = json.loads(json_data)["boundingBoxes"]

    try:
        image = Image.open(image_file).convert("RGB")
        print(np.array(image).shape)
        image.save("E:/Projects/API_Test/Model_API/api/kupa/tmp.png")
        # Do something with the image, e.g., save or process it

        # return jsonify({"message": "Image received successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    image_contours, json_outputs = [], []
    for box in bounding_boxes:
        temp_coords = box["coords"]
        x1, y1, x2, y2 = (
            temp_coords["x1"],
            temp_coords["y1"],
            temp_coords["x2"],
            temp_coords["y2"],
        )

        tmp_image = np.array(image)[y1:y2, x1:x2]
        mask = seg.make_segmentation(tmp_image)

        text_regions, coords = extract_text_regions(mask, np.array(image))

        extracted_contours, extracted_outputs = [], []

        for i in range(len(text_regions)):
            tmp_region = text_regions[i]
            decoded_region = ocr.make_transcription(tmp_region)

            extracted_contours.append(coords[i])
            extracted_outputs.append(decoded_region)

            print(decoded_region, coords)

        temp_json_output = create_output(extracted_outputs, extracted_contours)
        image_contours.append([x1, y1, x2, y2])
        json_outputs.append(temp_json_output)

        json_output = create_final_output(json_outputs, image_contours)

        return json_output, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
