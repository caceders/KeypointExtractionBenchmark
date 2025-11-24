import cv2
from benchmark.feature_extractor import FeatureExtractor
from benchmark.utils import load_HPSequences
from benchmark.feature import Feature
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from typing import Callable
import numpy as np
from benchmark.image_feature_set import ImageFeatureSet, ImageFeatureSequence, ImageFeatures
from benchmark.matching import Match, homographic_optimal_matching, greedy_maximum_bipartite_matching
import sys

## Set constants and configs
FAST = False
MAX_FEATURES = 500
DISTANCE_THRESHOLD = 40
SQUARED_DISTANCE_THRESHOLD = DISTANCE_THRESHOLD ** 2
DISTANCE_TYPE = cv2.NORM_L2

maching_approach: Callable[[list[Feature], list[Feature], int], list[Match]] = greedy_maximum_bipartite_matching
match_properties_for_mAP_calculation = ["distance", "average_response", "average_ratio"]

## Load dataset and setup feature extractor
img_seqs, hom_seq = load_HPSequences(r"hpatches-sequences-release")
SIFT = cv2.SIFT_create()
ORB = cv2.ORB_create()

keypoint_extractor = FeatureExtractor.from_opencv(SIFT.detect, SIFT.compute, True, 16, 16)

## Find features in all images
image_feature_set = ImageFeatureSet(len(img_seqs), len(img_seqs[0]))
for seq_idx, img_seq in enumerate(tqdm(img_seqs, leave=False, desc="Finding all features")):
    for img_idx, img in enumerate(img_seq):

        kps = keypoint_extractor.detect_kps(img)
        descs = keypoint_extractor.describe_kps(img, kps)
        features = [Feature(kp, desc) for _, (kp, desc) in enumerate(zip(kps, descs))]

        if FAST:
            features = features[:50]
        else:
            features = features[:MAX_FEATURES]
        
        image_feature_set[seq_idx][img_idx].add_feature(features)



## Calculate valid matches
for seq_idx, image_feature_sequence in enumerate(tqdm(image_feature_set, leave=False, desc="Calculating and ranking all valid matches")):
    for reference_feature in tqdm(image_feature_sequence.ref_image, leave=False):
        for related_image_idx, related_image in enumerate(image_feature_sequence.rel_images):
            for related_feature in related_image:
                transformed_related_feature_pt = related_feature.get_pt_after_homography_transform(hom_seq[seq_idx][img_idx - 1])
                dist_squared = (reference_feature.pt[0] - transformed_related_feature_pt[0])**2 + (reference_feature.pt[1] - transformed_related_feature_pt[1])**2
                
                if dist_squared > SQUARED_DISTANCE_THRESHOLD:
                    continue

                reference_feature.store_valid_match_for_image(related_image_idx, related_feature, dist_squared)
                related_feature.store_valid_match_for_image(0, reference_feature, dist_squared)



## Calculate repeatability based on optimal homographical matching
set_possible_correct_matches: list[list[int]] = []
set_repeatabilities: list[list[float]] = []
for seq_idx, image_feature_sequence in enumerate(tqdm(image_feature_set, leave=False, desc="Calculating optimal matching results")):

    num_possible_correct_matches = []
    repeatability = []
    
    for related_image_idx, related_image in enumerate(image_feature_sequence.rel_images):
        
        ref_features = image_feature_sequence.ref_image.get_features()
        rel_features = image_feature_sequence.rel_image(related_image_idx).get_features()

        matches = homographic_optimal_matching(ref_features, rel_features, hom_seq[seq_idx][related_image_idx-1])

        num_correct_matches = 0
        for match in matches:
            if match.score < DISTANCE_THRESHOLD:
                num_correct_matches += 1 

        num_possible_correct_matches.append(num_correct_matches)
        repeatability.append(num_correct_matches/len(ref_features))

    set_possible_correct_matches.append(num_possible_correct_matches)
    set_repeatabilities.append(repeatability)

for seq_idx in range(len(set_possible_correct_matches)):
    print()
    print(f"cm: mean {np.mean(set_possible_correct_matches[seq_idx])} standard_deviation: {np.std(set_possible_correct_matches[seq_idx])}")
    print(f"rep: mean {np.mean(set_repeatabilities[seq_idx])} standard_deviation: {np.std(set_repeatabilities[seq_idx])}")

set_possible_correct_matches = np.array(set_possible_correct_matches)
set_possible_correct_matches.flatten()

set_repeatabilities = np.array(set_repeatabilities)
set_repeatabilities.flatten()

print(f"cm_total: mean {np.mean(set_possible_correct_matches)} standard_deviation: {np.std(set_possible_correct_matches)}")
print(f"rep_total: mean {np.mean(set_repeatabilities)} standard_deviation: {np.std(set_repeatabilities)}")



## Do matching
num_correct_matches = []
all_matches = []
all_distance_AP = []
all_keypoint_response_AP = []
all_ratio_AP = []
for seq_idx, image_feature_sequence in enumerate(tqdm(image_feature_set, leave=False, desc="Calculating matching results")):
    sequence_matches = []
    for related_image_idx, related_image in enumerate(image_feature_sequence.rel_images):
        # Reference and related features
        ref_features = image_feature_sequence.ref_image.get_features()
        rel_features = image_feature_sequence.rel_image(related_image_idx).get_features()

        matches = maching_approach(ref_features, rel_features, DISTANCE_TYPE)
        sequence_matches.extend(matches)

    # Calculate APs

    sequence_matches.sort(key=lambda x: x.custom_properties["distance"])
    labels = [(1 if match.is_correct else 0) for match in sequence_matches]
    scores = [1/match.custom_properties["distance"] if match.custom_properties["distance"] != 0 else sys.float_info.max  for match in sequence_matches]
    sequence_distance_AP = average_precision_score(labels, scores)

    sequence_matches.sort(key=lambda x: x.custom_properties["average_response"])
    labels = [(1 if match.is_correct else 0) for match in sequence_matches]
    scores = [match.custom_properties["average_response"] for match in sequence_matches]
    sequence_keypoint_response_AP = average_precision_score(labels, scores)

    sequence_matches.sort(key=lambda x: x.custom_properties["average_ratio"])
    labels = [(1 if match.is_correct else 0) for match in sequence_matches]
    scores = [1/match.custom_properties["average_ratio"] if match.custom_properties["average_ratio"] != 0 else sys.float_info.max for match in sequence_matches]
    sequence_ratio_AP = average_precision_score(labels, scores)

    all_distance_AP.append(sequence_distance_AP)
    all_keypoint_response_AP.append(sequence_keypoint_response_AP)
    all_ratio_AP.append(sequence_ratio_AP)
    all_matches.append(sequence_matches)



print(f"total num matches: {sum(len(sequence_matches) for sequence_matches in all_matches)}")

total_possible_correct_matches = sum(
    num_correct_matches
    for num_correct_sequence_matches in set_possible_correct_matches
    for num_correct_matches in num_correct_sequence_matches
)

print(f"num possible correct matches: {total_possible_correct_matches}")

total_correct_matches = sum(
    1 if match.is_correct else 0
    for sequence_matches in all_matches
    for match in sequence_matches
)

print(f"num correct matches: {total_correct_matches}")


distance_mAP = np.average(all_distance_AP)
print(f"distance mAP: {distance_mAP}")

keypoint_response_mAP = np.average(all_keypoint_response_AP)
print(f"keypoint response mAP: {keypoint_response_mAP}")

ratio_mAP = np.average(all_ratio_AP)
print(f"ratio mAP: {ratio_mAP}")