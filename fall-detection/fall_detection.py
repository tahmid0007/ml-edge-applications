from roboflowoak import RoboflowOak
import cv2
import time
import numpy as np
import copy


def calculate_iou(box1, box2):
    x_left = max(box1['x'], box2['x'])
    y_top = max(box1['y'], box2['y'])
    x_right = min(box1['x'] + box1['width'], box2['x'] + box2['width'])
    y_bottom = min(box1['y'] + box1['height'], box2['y'] + box2['height'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    box1_area = box1['width'] * box1['height']
    box2_area = box2['width'] * box2['height']

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


def confirm_consecutive_detection(preds, DET_HISTORY, COUNTER):
    refined_preds = copy.deepcopy(preds)
    for idx, one_curr in enumerate(preds):
        flag = False
        for item_hist in DET_HISTORY[COUNTER - 2]:                 
            iou = calculate_iou(item_hist, one_curr)
            if iou > 0.8: 
                print("one box overlap found in history")
                flag = True

        if flag == False:
            print("deleting bcz no match in history ", refined_preds[idx])
            del refined_preds[idx]

    return refined_preds

def confirm_size(preds):
    refined_preds = copy.deepcopy(preds)
    for idx, one in enumerate(preds):
        if one["confidence"] < 0.5: continue
        area = one["width"] * one["height"]
        if area > 0.7 * (640 * 640): 
            print("deleting bcz size test failed ", refined_preds[idx])
            del refined_preds[idx]
    return refined_preds

def main():
        DET_HISTORY = []
        COUNTER = 0

        rf = RoboflowOak(
        model="fall-detection-ca3o8", 
        confidence=0.05, 
        overlap=0.5,
        version="4", 
        api_key="JUy2iq3uJkeSj9zB2Khx", rgb=True,
        device=None
        )

        while True:
            t0 = time.time()
            #time.sleep(0.1)
            result, frame, raw_frame, depth = rf.detect()
            predictions = result["predictions"]
            
            preds = [p.json() for p in predictions]
            #print("PREDICTIONS ", preds)

            refined_preds = confirm_size(preds)

            DET_HISTORY.append(refined_preds)

            if COUNTER > 3:
                refined_preds = confirm_consecutive_detection(refined_preds, DET_HISTORY, COUNTER)


            #cv2.imshow("frame", frame)

            COUNTER += 1
            t = time.time()-t0
            print("INFERENCE TIME IN MS ", 1/t)

            if cv2.waitKey(1) == ord('q'):
                break

if __name__ == '__main__':
    main()