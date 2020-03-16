# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo which runs object detection on camera frames.

export TEST_DATA=/usr/lib/python3/dist-packages/edgetpu/test_data

Run face detection model:
python3 -m edgetpuvision.detect \
  --model ${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite

Run coco model:
python3 -m edgetpuvision.detect \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt

Press Q key to exit.

"""
from CentroidTracker import CentroidTracker
import cv2
from PIL import Image
import argparse
import re
import os
from edgetpu.detection.engine import DetectionEngine
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import imutils
import time



def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
       lines = (p.match(line).groups() for line in f.readlines())
       return {int(num): text.strip() for num, text in lines}


def main():
    default_model_dir = '../all_models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=30,
                        help='number of classes with highest score to display')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='class score threshold')
    args = parser.parse_args()

    print("Loading %s with %s labels."%(args.model, args.labels))
    engine = DetectionEngine(args.model)
    labels = load_labels(args.labels)
    
    ct = CentroidTracker()
   
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2_im = frame
        height, width, channels = cv2_im.shape

        pil_im = Image.fromarray(cv2_im)

        objs = engine.DetectWithImage(pil_im, threshold=args.threshold,
                                    keep_aspect_ratio=True, relative_coord=True,
                                    top_k=args.top_k)

        rects = []

        for obj in objs:
          percent = int(100 * obj.score)
          label = '%d%% %s' % (percent, labels[obj.label_id])

          if 'bottle' in label and percent > 50:
            rects.append(obj.bounding_box)

            x0, y0, x1, y1 = obj.bounding_box.flatten().tolist()
            x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height) 

            cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2_im = cv2.putText(cv2_im, label[0:2], (x0, y0+30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        objects = ct.update(rects)
            
        if objects is not None:
          cv2_im = cv2.putText(cv2_im, str(len(rects)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
          c = 0
          for(objectID, centroid) in objects.items():
            c += 1
            #print("object: id: " , objectID)
           # print("centroid: " , centroid)
            text = "ID {}".format(objectID)
            cv2_im = cv2.putText(cv2_im, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2_im = cv2.circle(cv2_im, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)



        cv2.imshow('frame', cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

