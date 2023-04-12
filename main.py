import cv2
from utils import get_parking_spots_bboxes, empty_or_not
mask_path = 'mask.png'
video_path = 'data/parking.mp4'

mask = cv2.imread(mask_path, 0)
# mask = mask.reshape(2082, 3700)

cap = cv2.VideoCapture(video_path)
ret = True

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
# print(len(connected_components))

spots = get_parking_spots_bboxes(connected_components)
# print(spots[0])

spots_status = [None for j in spots]
step = 30

while ret:
    ret, frame = cap.read()
    for sopt_idx, spot in enumerate(spots):
        x1, y1, w, h = spot
        
        spot_crop = frame[y1:y1+h, x1:x1+h]

        spot_status = empty_or_not(spot_bgr=spot_crop)

        spots_status[sopt_idx] = spot_status

        if spot_status:
            cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), [0,255,0],3)
        else:
            cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), [0,0,255],3)

    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()