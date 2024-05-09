import numpy as np
from PIL import Image
import requests
import json


def send_request():
    p = "D:/Databases/06/forms_obciete/a01-000u.png"
    p = "C:/Users/Piotr/Desktop/IAM_img/a01-000u.png"
    # p = "C:/Users/Piotr/Desktop/IAM_img/imageimage3.png"
    # p = "C:/Users/Piotr/Desktop/IAM_img/a01-049.png"
    # p = "C:/Users/Piotr/Desktop/IAM_img/linie1.jpg"
    p = "C:/Users/Piotr/Desktop/IAM_img/test_image.png"

    with open(p, 'rb') as f:
        img = f.read()
    api_link = "http://localhost:5000/upload_image"

    bounding_boxes_data = {
        "boundingBoxes": [
            {
                "coords": {
                    "x1": 0,
                    "y1": 0,
                    "x2": 2390,
                    "y2": 1250
                }
            },
            {
                "coords": {
                    "x1": 0,
                    "y1": 1280,
                    "x2": 2390,
                    "y2": 2500
                }
            }
        ]
    }

    file = {'image': img}
    data = {'json': json.dumps(bounding_boxes_data),
            'token': "kupa"
            }
    response = requests.post(api_link, files=file, data=data)
    print(response.json())
    print(response.status_code)


def main():
    send_request()


if __name__ == "__main__":
    main()
