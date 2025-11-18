import cv2
from benchmark.feature_extractor import FeatureExtractor
from benchmark.utils import load_HPSequences
from benchmark.feature import Feature
from tqdm import tqdm
import numpy as np
from benchmark.image_feature_set import ImageFeatureSet, ImageFeatureSequence, ImageFeatures
from matching.games import StableMarriage

## Set constants and configs
FAST = True
OVERLAP_THRESHOLD = 0.4
DISTANCE_THRESHOLD = 10
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
            features = features[:100]
        
        image_feature_set[seq_idx][img_idx].add_features(features)



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
matcher = cv2.BFMatcher(DISTANCE_TYPE, crossCheck=False)
for seq_idx, image_feature_sequence in enumerate(tqdm(image_feature_set, leave=False, desc="Calculating optimal matching results")):

    num_possible_correct_matches = []
    repeatability = []
    
    for related_image_idx, related_image in enumerate(image_feature_sequence.rel_images):
        
        
        # Reference and related features
        ref_features = image_feature_sequence.ref_image.get_features()
        rel_features = image_feature_sequence.rel_image(related_image_idx).get_features()

        # Compute pairwise distance matrix
        ref_pts = np.array([f.pt for f in ref_features], dtype=np.float32)
        rel_pts = np.array([f.get_pt_after_homography_transform(hom_seq[seq_idx][related_image_idx])
                            for f in rel_features], dtype=np.float32)

        if len(ref_pts) == 0 or len(rel_pts) == 0:
            matches = []
        else:
            dists = np.linalg.norm(ref_pts[:, None, :] - rel_pts[None, :, :], axis=2)

            # Greedy one-to-one matching
            matches = []
            used_ref = set()
            used_rel = set()

            # Sort all pairs by distance
            pairs = [(i, j, dists[i, j]) for i in range(dists.shape[0]) for j in range(dists.shape[1])]
            pairs.sort(key=lambda x: x[2])

            for i, j, dist in pairs:
                if i not in used_ref and j not in used_rel:
                    matches.append((ref_features[i], rel_features[j], dist))
                    used_ref.add(i)
                    used_rel.add(j)


        matches.sort(key=lambda x: x[2])  # sort by distance

        correct_matches = 0
        for _, _, distance in matches:
            if distance < DISTANCE_THRESHOLD:
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