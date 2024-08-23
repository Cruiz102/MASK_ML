import torch.amp
import cv2
from typing import List
import torch
import numpy as np
from third_party.sam_cutie import SamCutiePipeline, BBoxOutput, DrawAnnotator, generate_random_rgb
import sys
import os



points = []
points_labels = []
objects_colors = {}
is_paused = False
pipeline = SamCutiePipeline()

def update_global_variables(event, x, y, flags, param):
   global points, points_labels
   if event == cv2.EVENT_LBUTTONDOWN:
       points.append((x,y))
       points_labels.append(1)
   elif event == cv2.EVENT_RBUTTONDOWN:
       points.append((x,y))
       points_labels.append(-1)


def update_objects(bbox: List[BBoxOutput]):
    global objects_colors
    for box in bbox:
        if box.track_id not in objects_colors:
            objects_colors[box.track_id] = generate_random_rgb()
@torch.inference_mode()
@torch.autocast("cuda")
def main():
    global is_paused, pipeline, points, objects_colors
    annotator = DrawAnnotator()
    cap = cv2.VideoCapture(2)
    names = {0: "Object"}
    cv2.namedWindow('demo', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("demo", update_global_variables)
    object_counter = 0
    while True:
        image = cap.read()
        if pipeline.objects:
            detected_bbox = pipeline.get_bounding_box(image)
            update_objects(detected_bbox)
            image = annotator.draw_bbox(image,detected_bbox,object_color=objects_colors ,names= names, draw_mask=True)     
        image_with_circles = image.copy()
        for i, point in enumerate(points):
            if points_labels[i] == 1:
                image_with_circles = cv2.circle(image_with_circles, point, 5, (255, 0, 0), 1)
            elif points_labels[-1] == -1:
                image_with_circles = cv2.circle(image_with_circles, point, 5, (0, 0, 255), 1)
        
        if len(points) > 1:
            cv2.imshow("demo", image_with_circles)  
        else:

            cv2.imshow("demo", image)

        key = cv2.waitKey(1)

        if key == ord("p"):
            is_paused ^= True
        if key == ord("a"):
            pipeline.save_object(f"object-{object_counter}", image, points=np.array(points),point_labels=points_labels)
            points.clear()
            points_labels.clear()
            object_counter += 1



if __name__ == '__main__':

    main()

