"""Click the corners of the bed on the first frame, press ENTER. Prints the polygon for config.yaml.
Usage: python tools/select_bed.py path/to/video.mp4"""
import sys, json
import cv2

video = sys.argv[1]
cap = cv2.VideoCapture(video)
ok, frame = cap.read()
assert ok, "cannot read video"
h, w = frame.shape[:2]
pts = []


def on_click(event, x, y, *_):
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))


cv2.namedWindow("bed"); cv2.setMouseCallback("bed", on_click)
while True:
    vis = frame.copy()
    for p in pts:
        cv2.circle(vis, p, 4, (0, 0, 255), -1)
    if len(pts) > 1:
        cv2.polylines(vis, [__import__("numpy").array(pts)], len(pts) > 2, (0, 200, 255), 2)
    cv2.imshow("bed", vis)
    k = cv2.waitKey(30) & 0xFF
    if k in (13, 10) and len(pts) >= 3:
        break
    if k == ord("u") and pts:
        pts.pop()
cv2.destroyAllWindows()
print("bed_polygon:")
print(json.dumps([[round(x / w, 4), round(y / h, 4)] for x, y in pts]))
