import cv2
import numpy as np
import os
import copy

def color_from_index(index):
    return ((index * 277)%256 + 100, (index * 377)%256, (index * 477)%256)

if not os.path.exists('./output'):
    os.makedirs('./output')

mask = cv2.imread('./images/mask.png', cv2.IMREAD_GRAYSCALE)

inverse_mask = 255 - mask
inverse_mask = cv2.copyMakeBorder(
    src=inverse_mask[1: -1, 1: -1],  # We sacrifice one line in the periphery because we do not want blobs touching the image border
    top=1,
    bottom=1,
    left=1,
    right=1,
    borderType=cv2.BORDER_CONSTANT,
    value=255
)


blob_detector_parameters = cv2.SimpleBlobDetector_Params()
blob_detector_parameters.filterByArea = False
blob_detector_parameters.filterByInertia = False
blob_detector_parameters.filterByConvexity = False
blob_detector_parameters.filterByCircularity = False


blob_detector = cv2.SimpleBlobDetector_create(blob_detector_parameters)

keypoints = blob_detector.detect(inverse_mask)
annotated_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
annotated_img = cv2.drawKeypoints(annotated_img, keypoints, annotated_img,
                                  color=(0, 0, 255),
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
"""for keypoint in keypoints:
    print(f"keypoint.pt = {keypoint.pt}")
    print(f"keypoint.angle = {keypoint.angle}")
    print(f"keypoint.class_id = {keypoint.class_id}")
    print(f"keypoint.octave = {keypoint.octave}")
    print(f"keypoint.response = {keypoint.response}")
    print(f"keypoint.size = {keypoint.size}")
"""
print(f"keypoints = {keypoints}")
print(f"type(keypoints) = {type(keypoints)}")

# For each keypoint, floodfill to get the blob surface
colored_blobs_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
running_ndx = 0

center_to_points = {}
center_to_boundingBox = {}
center_to_blobImg = {}
for keypoint in keypoints:
    center = (round(keypoint.pt[0]), (round(keypoint.pt[1])))
    inverse_mask_copy = cv2.cvtColor(copy.deepcopy(inverse_mask), cv2.COLOR_GRAY2BGR)
    if (inverse_mask_copy[center[1], center[0]]).all() == 0:  # We are in the blob
        cv2.floodFill(inverse_mask_copy, None,
                      seedPoint=center,
                      newVal=color_from_index(running_ndx))
    blob_mask = inverse_mask_copy - cv2.cvtColor(inverse_mask, cv2.COLOR_GRAY2BGR)
    grayscale_blob_mask = cv2.cvtColor(blob_mask, cv2.COLOR_BGR2GRAY)
    retval, grayscale_blob_mask = cv2.threshold(grayscale_blob_mask, 0, 255, cv2.THRESH_BINARY)
    center_to_blobImg[center] = grayscale_blob_mask
    # Get the non zero points
    non_zero_points_arr = cv2.findNonZero(cv2.cvtColor(blob_mask, cv2.COLOR_BGR2GRAY))
    center_to_points[center] = non_zero_points_arr
    # Blob bounding box
    bounding_box = cv2.boundingRect(non_zero_points_arr)
    print(f"bounding_box = {bounding_box}")
    center_to_boundingBox[center] = bounding_box

    cv2.imwrite(f'./output/blob_mask_{running_ndx}.png', blob_mask)
    colored_blobs_img = np.maximum(colored_blobs_img, blob_mask)
    cv2.rectangle(colored_blobs_img, bounding_box, (0, 255, 0), thickness=1)

    running_ndx += 1

# Extract blob moments
for center, blob_img in center_to_blobImg.items():
    print(f"center = {center}")
    moments = cv2.moments(blob_img)
    print(f"moments: \n{moments}")
    number_of_points = len(center_to_points[center])
    print(f"moments['m00']/255 = {moments['m00']/255}; number_of_points = {number_of_points}")
    centroid = (moments['m10']/moments['m00'], moments['m01']/moments['m00'])
    print(f"centroid = {centroid}")

cv2.imwrite('./output/inverse_mask.png', inverse_mask)
cv2.imwrite('./output/annotated.png', annotated_img)
cv2.imwrite('./output/colored_blobs.png', colored_blobs_img)
