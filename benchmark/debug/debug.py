from ..image_feature_set import ImageFeatureSet, ImageFeatureSequence
from ..feature import Feature
import cv2
import numpy as np

def display_feature_in_image(dataset_image_sequences: list, sequence_index: int, image_index: int, feature: Feature):
    image = dataset_image_sequences[sequence_index][image_index]

    keypoint = feature.keypoint

    lookimage=cv2.drawKeypoints(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),[keypoint],image,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow(f"sequence {sequence_index}", lookimage)
    cv2.waitKey(0)

def display_feature_for_sequence(dataset_image_sequences: list, sequence_index: int, image_feature_set: ImageFeatureSet):
    lookimages = []
    images = [img for img in dataset_image_sequences[sequence_index]]
    for image_index, image in enumerate(images):
        keypoints = [feature.keypoint for feature in image_feature_set[sequence_index][image_index]]

        lookimage=cv2.drawKeypoints(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),keypoints,image,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        lookimages.append(lookimage)
    grid_image = np.hstack(images)
    cv2.imshow(f"sequence {sequence_index}", grid_image)
    cv2.waitKey(0)