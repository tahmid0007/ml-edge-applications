from roboflowoak import RoboflowOak
import cv2
import time

if __name__ == '__main__':
    rf = RoboflowOak(model="fall-detection-ca3o8", confidence=0.05, overlap=0.5,
    version="4", api_key="JUy2iq3uJkeSj9zB2Khx", rgb=True, device=None)
    while True:
        result, frame, raw_frame, depth = rf.detect(visualize=True)
        predictions = result["predictions"]
        print("PREDICTIONS ", [p.json() for p in predictions])
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord('q'):
            break