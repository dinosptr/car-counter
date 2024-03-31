from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 720)

model = YOLO("model/yolov8n.pt")


while True:
  success, img = cap.read()
  results = model(img)

  for result in results:
    boxes = result.boxes
    for box in boxes:
      x1, y1, x2, y2 = box.xyxy[0]
      x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

      # cara pertama menggunakan library opencv (cv2)
      # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

      # Cara kedua menggunakan cvzone
      w, h = x2-x1, y2-y1
      cvzone.cornerRect(img, (x1, y1, w, h))

      # menambahkan confidence pada bbox
      conf = math.ceil((box.conf[0] * 100)) / 100
      cvzone.putTextRect(img, f'{conf}', (max(0, x1), max(35, y1)))
  cv2.imshow("Image", img)
  cv2.waitKey(1)