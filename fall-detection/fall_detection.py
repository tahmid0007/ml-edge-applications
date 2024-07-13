from roboflowoak import RoboflowOak
import cv2
import time
import numpy as np
import yaml
import os
from datetime import datetime
import telepot


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
    refined_preds = []
    for one_curr in preds:
        flag = False
        for item_hist in DET_HISTORY[COUNTER - cfg["lookback_frames"]]:
            iou = calculate_iou(item_hist, one_curr)
            if iou > cfg["lookback_iou"]:
                print("one box overlap found in history")
                flag = True
                break

        if flag:
            refined_preds.append(one_curr)

    return refined_preds

def confirm_size(preds):
    refined_preds = []
    for one in preds:
        area = one["width"] * one["height"]
        if area <= cfg["size_discard_coeff"] * (640 * 640):
            refined_preds.append(one)
        else:
            print("deleting bcz size test failed ", one)
    return refined_preds

def create_output_folder():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_path = os.path.join("/home/tahmid/Desktop/datasets/detections/", timestamp)
    os.makedirs(folder_path)
    return folder_path

def alert(image_path, message = "fall_detected"):
    bot = telepot.Bot(cfg["telegram_api_key"])
    bot.sendMessage(cfg["telegram_chat_id"], message)
    #bot.sendPhoto(cfg["telegram_chat_id"], photo=open(image_path, 'rb'))

def main():
    folder_path = create_output_folder()
    DET_HISTORY = []
    COUNTER = 0

    while True:
        if COUNTER == 1000:
            COUNTER = 0
            DET_HISTORY = []

        t0 = time.time()
        result, frame, raw_frame, depth = rf.detect()
        predictions = result["predictions"]
        
        preds = [p.json() for p in predictions]

        refined_preds = confirm_size(preds)

        DET_HISTORY.append(refined_preds)

        if COUNTER > 3:
            refined_preds = confirm_consecutive_detection(refined_preds, DET_HISTORY, COUNTER)

        for bbox in refined_preds:
            print("detected fall....", refined_preds)
            x1, y1, x2, y2 = bbox['x'] - bbox['width'] / 2, bbox['y'] - bbox['height'] / 2,  bbox['x'] + bbox['width'] / 2, bbox['y'] + bbox['height'] / 2
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(raw_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            image_path = folder_path + "/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%Ms") + ".jpg"
            cv2.imwrite(image_path, raw_frame[y1: y2, x1:x2])

        cv2.imshow("frame", raw_frame)

        if len(refined_preds) == 0:
            print("no fall event...")
        else:
            alert(image_path)
            
        COUNTER += 1
        t = time.time() - t0
        print("INFERENCE TIME IN SECONDS", t)

        time.sleep(cfg["sleep_time"])

        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == "__main__":

    path = os.path.dirname(os.path.abspath(__file__)) + "/files/cfg.yaml"
    with open(path, "r") as yf:
        cfg = yaml.safe_load(yf)

    rf = RoboflowOak(
        model = cfg["model"], 
        confidence = cfg["main_conf"], 
        overlap = cfg["nms_overlap"],
        version="4", 
        api_key = cfg["api_key"], rgb=True,
        device=None
    )

    main()
