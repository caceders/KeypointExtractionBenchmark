import cv2
from benchmark.feature_extractor import FeatureExtractor
from benchmark.utils import load_HPSequences
from benchmark.feature import Feature
from tqdm import tqdm
from sklearn.metrics import average_precision_score
import numpy as np
from benchmark.image_feature_set import ImageFeatureSet, ImageFeatureSequence, ImageFeatures
from benchmark.matching import Match, homographic_optimal_matching, greedy_maximum_bipartite_matching

## Set constants and configs
FAST = False
OVERLAP_THRESHOLD = 0.4
DISTANCE_THRESHOLD = 20
SQUARED_DISTANCE_THRESHOLD = 10 * 2
DISTANCE_TYPE = cv2.NORM_HAMMING


## Load dataset and setup feature extractor
img_seqs, hom_seq = load_HPSequences(r"hpatches-sequences-release")
ORB = cv2.ORB_create()
keypoint_extractor = FeatureExtractor.from_opencv(ORB.detect, ORB.compute)



## Find features in all images
image_feature_set = ImageFeatureSet(len(img_seqs), len(img_seqs[0]))
for seq_idx, img_seq in enumerate(tqdm(img_seqs, leave=False, desc="Finding all features")):
    for img_idx, img in enumerate(img_seq):

        kps = keypoint_extractor.detect_kps(img)
        descs = keypoint_extractor.describe_kps(img, kps)
        features = [Feature(kp, desc) for _, (kp, desc) in enumerate(zip(kps, descs))]

        if FAST:
            features = features[:50]
        
        image_feature_set[seq_idx][img_idx].add_feature(features)



## Calculate valid matches
i = 0
for seq_idx, image_feature_sequence in enumerate(tqdm(image_feature_set, leave=False, desc="Calculating and ranking all valid matches")):
    for reference_feature in tqdm(image_feature_sequence.ref_image, leave=False):
        for related_image_idx, related_image in enumerate(image_feature_sequence.rel_images):
            for related_feature in related_image:
                i += 1
                transformed_related_feature_pt = related_feature.get_pt_after_homography_transform(hom_seq[seq_idx][img_idx - 1])
                dist_squared = (reference_feature.pt[0] - transformed_related_feature_pt[0])**2 + (reference_feature.pt[1] - transformed_related_feature_pt[1])**2
                
                if dist_squared > SQUARED_DISTANCE_THRESHOLD:
                    continue

                reference_feature.store_valid_match_for_image(related_image_idx, related_feature, dist_squared)
                related_feature.store_valid_match_for_image(0, reference_feature, dist_squared)
print(i)



## Calculate repeatability based on optimal homographical matching
nums_possible_correct_matches: list[list[int]] = []
repeatabilities: list[list[float]] = []
for seq_idx, image_feature_sequence in enumerate(tqdm(image_feature_set, leave=False, desc="Calculating optimal matching results")):

    num_possible_correct_matches = []
    repeatability = []
    
    for related_image_idx, related_image in enumerate(image_feature_sequence.rel_images):
        
        
        # Reference and related features
        ref_features = image_feature_sequence.ref_image.get_features()
        rel_features = image_feature_sequence.rel_image(related_image_idx).get_features()

        matches = homographic_optimal_matching(ref_features, rel_features, hom_seq[seq_idx][related_image_idx-1])

        correct_matches = 0
        for match in matches:
            if match.score < DISTANCE_THRESHOLD:
                correct_matches += 1 

        num_possible_correct_matches.append(correct_matches)
        repeatability.append(correct_matches/len(ref_features))

    nums_possible_correct_matches.append(num_possible_correct_matches)
    repeatabilities.append(repeatability)

for seq_idx in range(len(nums_possible_correct_matches)):
    print()
    print(f"cm: mean {np.mean(nums_possible_correct_matches[seq_idx])} standard_deviation: {np.std(nums_possible_correct_matches[seq_idx])}")
    print(f"rep: mean {np.mean(repeatabilities[seq_idx])} standard_deviation: {np.std(repeatabilities[seq_idx])}")

nums_possible_correct_matches = np.array(nums_possible_correct_matches)
nums_possible_correct_matches.flatten()

repeatabilities = np.array(repeatabilities)
repeatabilities.flatten()

print(f"cm_total: mean {np.mean(nums_possible_correct_matches)} standard_deviation: {np.std(nums_possible_correct_matches)}")
print(f"rep_total: mean {np.mean(repeatabilities)} standard_deviation: {np.std(repeatabilities)}")



## Do matching
num_correct_matches = []
all_matches = []
all_AP = []
for seq_idx, image_feature_sequence in enumerate(tqdm(image_feature_set, leave=False, desc="Calculating matching results")):
    sequence_matches = []
    for related_image_idx, related_image in enumerate(image_feature_sequence.rel_images):
        # Reference and related features
        ref_features = image_feature_sequence.ref_image.get_features()
        rel_features = image_feature_sequence.rel_image(related_image_idx).get_features()

        matches = greedy_maximum_bipartite_matching(ref_features, rel_features)
        sequence_matches.extend(matches)

    labels = [(1 if match.is_correct else 0) for match in sequence_matches]
    scores = [match.score for match in sequence_matches]

    sequence_AP = average_precision_score(labels, scores)
    all_AP.append(sequence_AP)
    all_matches.append(sequence_matches)



print(f"num matches: {sum(len(sequence_matches) for sequence_matches in all_matches)}")
mAP = np.average(all_AP)

print(f"mAP: {mAP}")