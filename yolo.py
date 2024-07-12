import ultralytics
from ultralytics import YOLO
ultralytics.checks()
import cv2
import numpy as np
import pytesseract as ptsr
import imutils




ptsr.pytesseract.tesseract_cmd = r'C:\Users\vhnam\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
info = []
color = f"defautl"

def TachNen(frame, mask_red, mask_green, mask_yellow):
    # Kết hợp các mặt nạ để giữ lại các vùng màu cần thiết
    combined_mask = cv2.bitwise_or(mask_yellow, mask_red)
    combined_mask = cv2.bitwise_or(combined_mask, mask_green)

    # Đảo ngược mặt nạ để xác định các vùng đen cần chuyển sang màu trắng
    inverse_mask = cv2.bitwise_not(combined_mask)

    # Chuyển các vùng đen sang màu trắng trên ảnh gốc
    frame[inverse_mask > 0] = [255, 255, 255]

    return frame

def expand_color(image, mask, color_bgr):
    expand_size = 5
    points = np.column_stack(np.where(mask > 0))
    for point in points:
        x, y = point
        for dx in range(-expand_size, expand_size + 1):
            for dy in range(-expand_size, expand_size + 1):
                if 0 <= x + dx < image.shape[0] and 0 <= y + dy < image.shape[1]:
                    image[x + dx, y + dy] = color_bgr

def extract_numbers_from_image(img):
    # Đọc hình ảnh sử dụng OpenCV
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = img

    # Kiểm tra kích thước hình ảnh và thay đổi kích thước nếu cần thiết
    # w, h, _ = img.shape
    # if w > 1500 and h > 1500:
    #     new_w, new_h = int(w * 0.25), int(h * 0.25)
    #     img = cv2.resize(img, (new_w, new_h))
    #     img = img.reshape(new_w, new_h, 3)

    boxes = ptsr.image_to_data(img)
    for x, b in enumerate(boxes.splitlines()):
        if x != 0:
            b = b.split()
            if b[len(b) - 1] != -1:
                #print(b)
                index_num = len(b)
                x, y, w, h = int(b[index_num - 6]), int(b[index_num - 5]), int(b[index_num - 4]), int(b[index_num - 3])
                cv2.rectangle(img, (x,y), (x+w, h+y), (0, 0, 255), 2)
                cv2.putText(img, b[index_num - 1], (x, y),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (50, 50, 255), 2)

    #cv2.imshow('ocr', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return b[index_num - 1]

def detection_color(image):
    # Đọc ảnh tĩnh từ tập tin
#    image_path = r'D:\TaiLieu\Do_An\Do_An_4\test_thucte\red2.jpg'
#   image_path = r'D:\TaiLieu\Do_An\Do_An_4\test_thucte\green_color.jpg'
#    image_path = r'D:\TaiLieu\Do_An\Do_An_4\test_thucte\red_color.jpg'
#    image_path = r'D:\TaiLieu\Do_An\Do_An_4\test_thucte\yellow.jpg'
#    image_path = r'D:\TaiLieu\Do_An\Do_An_4\test_thucte\yellow2.jpg'
#    image_path = r'D:\TaiLieu\Do_An\Do_An_4\test_thucte\test1.jpg'
#    image_path = r'D:\TaiLieu\Do_An\Do_An_4\test_thucte\test2.jpg'
#    image_path = r'D:\TaiLieu\Do_An\Do_An_4\test_thucte\test3.jpg'
#    image_path = r'D:\TaiLieu\Do_An\Do_An_4\test_thucte\test4.jpg'
#    image_path = r'D:\TaiLieu\Do_An\Do_An_4\test_thucte\test5.jpg'
    frame = image

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    lower_green = np.array([80, 100, 100])
    upper_green = np.array([120, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    frame = TachNen(frame, mask_red, mask_green, mask_yellow)

    expand_color(frame, mask_red, [0, 0, 255])
    expand_color(frame, mask_yellow, [0, 255, 255])
    expand_color(frame, mask_green, [0, 255, 0] )

    contours_red = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_red = imutils.grab_contours(contours_red)

    contours_green = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_green = imutils.grab_contours(contours_green)

    contours_yellow = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow = imutils.grab_contours(contours_yellow)

    color = f"defautl"
    for cnt in contours_red:
        area = cv2.contourArea(cnt)
        if area > 400:
            x, y, w, h = cv2.boundingRect(cnt)
            # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            # cv2.putText(frame, 'Red', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            color = 'Red'

    for cnt in contours_green:
        area = cv2.contourArea(cnt)
        if area > 400:
            x, y, w, h = cv2.boundingRect(cnt)
            # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # cv2.putText(frame, 'Green', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            color = 'Green'

    for cnt in contours_yellow:
        area_yellow = cv2.contourArea(cnt)
        if area_yellow > 200:
            x, y, w, h = cv2.boundingRect(cnt)
            #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            #cv2.putText(frame, 'Yellow', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            color = 'Yellow'

    if color == 'defautl' :
        txt = ' '
        color = ' '
    elif color == 'Red':
        #cv2.imshow('mask_red', mask_red)
        txt = extract_numbers_from_image(mask_red)
    elif color == 'Green':
        #cv2.imshow('mask_green', mask_green)
        txt = extract_numbers_from_image(mask_green)
    else:
        #cv2.imshow('mask_yellow', mask_yellow)
        txt = extract_numbers_from_image(mask_yellow)

    mask_green_colored = cv2.cvtColor(mask_green, cv2.COLOR_GRAY2BGR)
    image_from_mask = mask_green_colored
    #cv2.imshow("clolor", image_from_mask)
    #cv2.imshow('Color Recognition Output', frame)
    print(f'Color: {color},  Text: {txt}')
    cv2.waitKey(0)
    # if k == 27:
    #     break

    cv2.destroyAllWindows()
    return f"{color} - {txt}"

def detec_yolo(img_path):
    model = YOLO("yolov8s.yaml")  # Build a new model from YAML
    model = YOLO("yolov8s.pt")
    result = model.predict(source=img_path)
    return (result)

def __main__():
#    img_path = r'D:\TaiLieu\Do_An\Do_An_4\test_thucte\test1.jpg'
#    img_path = r'D:\TaiLieu\Do_An\Do_An_4\test_thucte\test2.jpg'
#    img_path = r'D:\TaiLieu\Do_An\Do_An_4\test_thucte\test3.jpg'
#    img_path = r'D:\TaiLieu\Do_An\Do_An_4\test_thucte\test4.jpg'
#    img_path = r'D:\TaiLieu\Do_An\Do_An_4\test_thucte\test4.jpg'
#    img_path = r'D:\TaiLieu\Do_An\Do_An_4\test_thucte\test5.jpg'
    img_path = r'D:\TaiLieu\Do_An\Do_An_4\test_thucte\test6.jpg'
    results = detec_yolo(img_path)

    if isinstance(results, list):
        # Xử lý từng mục trong danh sách
        for result in results:
            # Lấy các bounding boxes từ kết quả
            boxes = result.boxes
            # Lấy nhãn của các bounding boxes
            labels = boxes.cls.numpy()
            # Lấy các tọa độ của bounding boxes
            coords = boxes.xyxy.numpy()
            # Lấy tên các nhãn từ kết quả
            names = result.names
            traffic_light_index = [i for i, name in names.items() if name == 'traffic light'][0]

            # Lọc các bounding boxes tương ứng với traffic light
            traffic_light_boxes = coords[labels == traffic_light_index]
            # Hiển thị thông tin các bounding boxes của traffic light
            print("Traffic Light Bounding Boxes:")
            for i, box in enumerate(traffic_light_boxes):
                #print(traffic_light_boxes)
                print(f"Box {i + 1}: {box}")
                x1, y1, x2, y2 = map(int, box)
                cropped_img = result.orig_img[y1:y2, x1:x2]
                img_crop2 = cropped_img.copy()
                info.append(detection_color(img_crop2))

            # Tạo một bản sao của hình ảnh gốc để vẽ các bounding boxes
            image_with_boxes = result.orig_img.copy()
            for i, box in enumerate(traffic_light_boxes):
                print(f'img {i+1} text: {info[i]} ')
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cv2.putText(image_with_boxes, str(info[i]), (x1, y1 -10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    # Màu xanh lá cây cho traffic light

                # Hiển thị hình ảnh với các bounding boxes
            #cv2.imshow('Traffic Lights', image_with_boxes)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

    else:
        # Nếu kết quả không phải là danh sách, xử lý như một kết quả duy nhất
        result = results
        # Lấy các bounding boxes từ kết quả
        boxes = result.boxes
        # Lấy nhãn của các bounding boxes
        labels = boxes.cls.numpy()
        # Lấy các tọa độ của bounding boxes
        coords = boxes.xyxy.numpy()
        # Lấy tên các nhãn từ kết quả
        names = result.names
        traffic_light_index = [i for i, name in names.items() if name == 'traffic light'][0]

        # Lọc các bounding boxes tương ứng với traffic light
        traffic_light_boxes = coords[labels == traffic_light_index]
        # Hiển thị thông tin các bounding boxes của traffic light
        print("Traffic Light Bounding Boxes:")
        for i, box in enumerate(traffic_light_boxes):
            # print(traffic_light_boxes)
            print(f"Box {i + 1}: {box}")
            x1, y1, x2, y2 = map(int, box)
            cropped_img = result.orig_img[y1:y2, x1:x2]
            img_crop2 = cropped_img.copy()
            info.append(detection_color(img_crop2))

        # Tạo một bản sao của hình ảnh gốc để vẽ các bounding boxes
        image_with_boxes = result.orig_img.copy()
        for i, box in enumerate(traffic_light_boxes):
            print(f'img {i + 1} text: {info[i]} ')
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_with_boxes, str(info[i]), (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # Màu xanh lá cây cho traffic light

            # Hiển thị hình ảnh với các bounding boxes
    cv2.imshow('Traffic Lights', image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __main__():
    __main__()