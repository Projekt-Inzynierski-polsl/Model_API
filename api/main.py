from flask import Flask, request, jsonify
import numpy as np
import json
import cv2
import requests
import os

import segmentation
import inference

app = Flask(__name__)


def convert_to_dict(coords):
    coords_dicts = []
    for coords in coords:
        coords_dicts.append({
            "leftX": coords[0],
            "leftY": coords[1],
            "rightX": coords[2],
            "rightY": coords[3]
        })
    return coords_dicts


def create_output(outputs, coords):
    coords_dicts = convert_to_dict(coords)

    decoded_output = []
    for i in range(len(outputs)):
        decoded_output.append({
            'content': str(outputs[i][0]),
            'leftX': coords_dicts[i]['leftX'],
            'leftY': coords_dicts[i]['leftY'],
            'rightX': coords_dicts[i]['rightX'],
            'rightY': coords_dicts[i]['rightY']
            # 'coordinates': coords_dicts[i]
        })
    # return jsonify({'decoded outputs': decoded_output})
    return decoded_output


def create_final_output(outputs, coords):
    coords_dicts = convert_to_dict(coords)

    decoded_output = []
    for i in range(len(coords)):
        decoded_output.append({
            # 'coordinates': coords_dicts[i],
            'leftX': coords_dicts[i]['leftX'],
            'leftY': coords_dicts[i]['leftY'],
            'rightX': coords_dicts[i]['rightX'],
            'rightY': coords_dicts[i]['rightY'],
            'lines': outputs[i]
        })
    return jsonify(decoded_output)


@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    # zapytaj Kube jak to ma wyglądać dokładnie

    token_field = request.form.get('token')
    # response = requests.post(request.remote_addr + "/api/acount/token",
    #                          headers={'Authorization': f'Bearer {token_field}'})

    response = requests.post("http://ocr-api:8080/api/account/token",
                             headers={'Authorization': f'Bearer {token_field}'})
    if response.status_code != 200:
        return 'Authorization failed', 400

    image_file = request.files['image']

    json_data = request.form.get('json')
    bounding_boxes = json.loads(json_data)['boundingBoxes']

    image_data = image_file.read()
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    num_channels = img.shape[2] if len(img.shape) == 3 else 1
    if num_channels == 1:
        grayscale_image = img
    else:
        grayscale_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    _, binary_image = cv2.threshold(grayscale_image, 150, 255, cv2.THRESH_BINARY_INV)

    image_contours, json_outputs = [], []
    for box in bounding_boxes:
        temp_coords = box['coords']
        x1, y1, x2, y2 = temp_coords['x1'], temp_coords['y1'], temp_coords['x2'], temp_coords['y2']

        temp_binary_image = binary_image[y1:y2, x1:x2]
        temp_segmented_image = segmentation.segment_image(temp_binary_image)

        contours, _ = cv2.findContours(temp_segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        extracted_lines, extracted_contours, extracted_outputs = [], [], []

        for i, contour in enumerate(contours):
            contour_mask = np.zeros_like(temp_segmented_image)
            cv2.drawContours(contour_mask, [contour], 0, 255, -1)
            text_line = cv2.bitwise_and(temp_binary_image, temp_binary_image, mask=contour_mask)
            x, y, w, h = cv2.boundingRect(contour)
            text_line_cropped = text_line[y:y + h, x:x + w]
            if cv2.contourArea(contour) >= 1500:
                extracted_lines.append(text_line_cropped)
                extracted_contours.append([x, y, x + w, y + h])
                output = inference.call_model(text_line_cropped)
                extracted_outputs.append(output)

        temp_json_output = create_output(extracted_outputs, extracted_contours)
        image_contours.append([x1, y1, x2, y2])
        json_outputs.append(temp_json_output)

    json_output = create_final_output(json_outputs, image_contours)

    return json_output, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)



{'image output': [{'coords': {'x1': 0, 'x2': 1800, 'y1': 0, 'y2': 378}, 'output': {'decoded outputs': [{'coords': {'x1': 221, 'x2': 1491, 'y1': 281, 'y2': 372}, 'text': ['industrie|,|"|Mr.|Brown|commented|icily|.|"|letus|have|a']}, {'coords': {'x1': 213, 'x2': 1491, 'y1': 158, 'y2': 249}, 'text': ['childrens|and|sick|people|than|to|take|onthis|vast']}, {'coords': {'x1': 205, 'x2': 1483, 'y1': 26, 'y2': 121}, 'text': ['|"Mr.|Powell|finds|iteasier|to|take|itout|of|mothers|,']}]}}]}