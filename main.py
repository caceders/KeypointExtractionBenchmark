from benchmark.debug import display_feature_for_sequence, display_feature_in_image
from benchmark.feature import Feature
from benchmark.feature_extractor import FeatureExtractor
from benchmark.image_feature_set import ImageFeatureSet, ImageFeatureSequence, ImageFeatures
from benchmark.matching import Match, MatchSet, MatchingApproach, MatchRankProperty,homographic_optimal_matching, greedy_maximum_bipartite_matching
from benchmark.utils import load_HPSequences
from tqdm import tqdm
import cv2
import math
import numpy as np
import random
import warnings



###################################### SETUP TESTBENCH HERE #################################################################

## Set constants and configs.
MAX_FEATURES = 200
RELATIVE_SCALE_DIFFERENCE_THRESHOLD = 1
DISTANCE_THRESHOLD = 10
DISTANCE_TYPE = cv2.NORM_L2 # cv2.NORM_L2 | cv2.NORM_HAMMING
VERIFICATION_CORRECT_TO_RANDOM_RATIO = 5
RETRIEVAL_CORRECT_TO_RANDOM_RATIO = 4000

## Load dataset.
dataset_image_sequences, dataset_homography_sequence = load_HPSequences(r"hpatches-sequences-release")

## Setup feature extractor.
SIFT = cv2.SIFT_create()

feature_extractor = FeatureExtractor.from_opencv(SIFT.detect, SIFT.compute, False)

## Setup matching approach
distance_match_rank_property = MatchRankProperty("distance", False)
average_response_match_rank_property = MatchRankProperty("average_response", True)
average_ratio_match_rank_property = MatchRankProperty("average_ratio", False)
mathing_properties = [distance_match_rank_property, average_response_match_rank_property, average_ratio_match_rank_property]

matching_approach = MatchingApproach(greedy_maximum_bipartite_matching, mathing_properties)

#############################################################################################################################

num_sequences = len(dataset_image_sequences)
num_related_images = len(dataset_image_sequences[0]) - 1


## Find features in all images.
image_feature_set = ImageFeatureSet(num_sequences, num_related_images)
for sequence_index, image_sequence in enumerate(tqdm(dataset_image_sequences, leave=False, desc="Finding all features")):
    for image_index, image in enumerate(image_sequence):

        keypoints = feature_extractor.detect_keypoints(image)
        descriptions = feature_extractor.describe_keypoints(image, keypoints)
        
        features = [Feature(keypoint, description, sequence_index, image_index)
                    for _, (keypoint, description)
                    in enumerate(zip(keypoints, descriptions))]

        features.sort(key= lambda x: x.keypoint.response)
        
        if MAX_FEATURES < len(features):
            features = random.sample(features, MAX_FEATURES)
        
        image_feature_set[sequence_index][image_index].add_feature(features)

        keypoints = [feature.keypoint for feature in features]
        descriptions = [feature.description for feature in features]


## Calculate valid matches.
for sequence_index, image_feature_sequenceuence in enumerate(tqdm(image_feature_set, leave=False, desc="Calculating and ranking all valid matches")):
    for reference_feature in tqdm(image_feature_sequenceuence.reference_image, leave=False):
        for related_image_index, related_image in enumerate(image_feature_sequenceuence.related_images):
            img_size = len(dataset_image_sequences[sequence_index][related_image_index+1])
            for related_feature in related_image:

                homography = dataset_homography_sequence[sequence_index][related_image_index]
                transformed_related_feature_pt = related_feature.get_pt_after_homography_transform(homography)
                distance = math.dist(reference_feature.pt, related_feature.pt)
                
                if distance > (DISTANCE_THRESHOLD * feature_extractor.get_description_image_scale_factor(img_size)):
                    continue

                reference_feature_size = reference_feature.keypoint.size
                related_feature_size = related_feature.get_size_after_homography_transform(homography)

                biggest_keypoint = max(reference_feature_size, related_feature_size)
                smallest_keypoint = min(reference_feature_size, related_feature_size)

                relative_scale_differnce = abs(1 - biggest_keypoint/smallest_keypoint)
                if relative_scale_differnce > RELATIVE_SCALE_DIFFERENCE_THRESHOLD:
                    continue

                reference_feature.store_valid_match_for_image(related_image_index, related_feature, distance)
                related_feature.store_valid_match_for_image(0, reference_feature, distance)



###############################################################################
# all_features = image_feature_set.get_features()
# all_sizes = np.array([feature.keypoint.size for feature in all_features])
# all_octaves = np.array([feature.keypoint.octave for feature in all_features])
# num_valid_matches = np.array([len(feature._all_valid_matches) for feature in all_features])
# highest_feature = max(all_features, key=lambda feature: len(feature._all_valid_matches))


# print(f"keypoint octave: max {max(all_octaves)}, min {min(all_octaves)}, mean {all_octaves.mean()} std {all_octaves.std()}")
# print(f"keypoint size: max {max(all_sizes)}, min {min(all_sizes)}, mean {all_sizes.mean()} std {all_sizes.std()}")
# print(f"valid matches: max {max(num_valid_matches)}, mean {num_valid_matches.mean()} std {num_valid_matches.std()}")

# display_feature_in_image(dataset_image_sequences, highest_feature.sequence_index, highest_feature.image_index, highest_feature)
# display_feature_for_sequence(dataset_image_sequences, highest_feature.sequence_index, image_feature_set)
###############################################################################



## Calculate repeatability and number of possible matches.
set_nums_possible_correct_matches: list[list[int]] = []
set_repeatabilities: list[list[float]] = []
for sequence_index, image_feature_sequenceuence in enumerate(tqdm(image_feature_set, leave=False, desc="Calculating optimal matching results")):

    nums_possible_correct_matches = []
    repeatabilities = []
    
    for related_image_index, related_image in enumerate(image_feature_sequenceuence.related_images):

        img_size = len(dataset_image_sequences[sequence_index][related_image_index+1])
        
        reference_features = image_feature_sequenceuence.reference_image.get_features()
        related_features = image_feature_sequenceuence.related_image(related_image_index).get_features()

        homography = dataset_homography_sequence[sequence_index][related_image_index]

        matches = homographic_optimal_matching(reference_features, related_features, homography) 

        # Check which matches were correct
        num_possible_correct_matches = 0
        for match in matches:
            if match.score < DISTANCE_THRESHOLD * feature_extractor.get_description_image_scale_factor(img_size):

                feature1_size = match.feature1.keypoint.size
                feature2_size = match.feature2.get_size_after_homography_transform(homography)

                biggest_keypoint = max(feature1_size, feature2_size)
                smallest_keypoint = min(feature1_size, feature2_size)

                relative_scale_differnce = 1 - biggest_keypoint/smallest_keypoint
                if relative_scale_differnce < RELATIVE_SCALE_DIFFERENCE_THRESHOLD:
                    num_possible_correct_matches += 1 

        repeatability = num_possible_correct_matches/len(reference_features)

        nums_possible_correct_matches.append(num_possible_correct_matches)
        repeatabilities.append(repeatability)

    set_nums_possible_correct_matches.append(nums_possible_correct_matches)
    set_repeatabilities.append(repeatabilities)



## Calculate matching results.
matching_match_set = MatchSet(num_sequences)

for sequence_index, image_feature_sequenceuence in enumerate(tqdm(image_feature_set, leave=False, desc="Calculating matching results")):
    for related_image_index, related_image in enumerate(image_feature_sequenceuence.related_images):

        # Reference and related features.
        reference_features = image_feature_sequenceuence.reference_image.get_features()
        related_features = image_feature_sequenceuence.related_image(related_image_index).get_features()

        matches = matching_approach.matching_callback(reference_features, related_features, DISTANCE_TYPE)
        matching_match_set[sequence_index].add_match(matches)




## Calculate verification results.
verification_match_set = MatchSet(len(dataset_image_sequences))

for sequence_index, image_feature_sequenceuence in enumerate(tqdm(image_feature_set, leave=False, desc="Calculating verification results")):
    reference_features = image_feature_sequenceuence.reference_image.get_features()
    for refrence_feature in reference_features:

        
        # Find related images with equivalent feature
        related_images_to_use = []

        for image_index in range(len(image_feature_sequenceuence.related_images)):
            if reference_feature.get_valid_matches_for_image(image_index):
                related_images_to_use.append(image_index)
        
        num_random_images = len(related_images_to_use) * VERIFICATION_CORRECT_TO_RANDOM_RATIO

        # Match for all relevant related images
        for related_image_index in range(len(related_images_to_use)):

            # Reference and related features.
            related_features = image_feature_sequenceuence.related_image(related_image_index).get_features()

            matches = matching_approach.matching_callback([reference_feature], related_features, DISTANCE_TYPE)
            verification_match_set[sequence_index].add_match(matches)
        
        # Pick random images
        choice_pool = [] #(sequence index, image index)
        for choice_sequence_index in range(len(image_feature_set)):
            if choice_sequence_index == sequence_index: # Do not take images from this sequence
                continue
            choice_pool.extend([(sequence_index, choice_image_index)
                                for choice_image_index
                                in range(len(image_feature_set[choice_sequence_index]))])
        
        chosen_random_images = random.sample(choice_pool, num_random_images)
        
        # Match for all random images
        for random_sequence_index, random_image_index in chosen_random_images:
            random_image_features = image_feature_set[random_sequence_index][image_index].get_features()
            matches = matching_approach.matching_callback([reference_feature], random_image_features, DISTANCE_TYPE)
            verification_match_set[sequence_index].add_match(matches)



## Calculate retrieval results.
retrieval_match_set = MatchSet(len(dataset_image_sequences))

for sequence_index, image_feature_sequenceuence in enumerate(tqdm(image_feature_set, leave=False, desc="Calculating retrieval results")):
    reference_features = image_feature_sequenceuence.reference_image.get_features()
    this_image_feature = image_feature_sequenceuence.reference_image
    all_features_except_this_image = [feature 
                                          for choice_image_feature_sequence in image_feature_set
                                          for choice_image_feature in choice_image_feature_sequence
                                          if choice_image_feature != this_image_feature
                                          for feature in choice_image_feature.get_features()]
    
    for refrence_feature in tqdm(reference_features, leave=False):
        
        correct_features = reference_feature.get_all_valid_matches()
        num_random_features = len(correct_features) * RETRIEVAL_CORRECT_TO_RANDOM_RATIO
        
        # Pick random features
        
        if num_random_features > len(all_features_except_this_image):
            warnings.warn(f"Not enough features to fullu calculate retrieval, need {num_random_features}, have {len(all_features_except_this_image)}. Reduce the acceptance threshold or increase feature count")

        chosen_random_features = random.sample(all_features_except_this_image, num_random_features)
        features_to_chose_from = correct_features + chosen_random_features
        
        # Match
        match = matching_approach.matching_callback([reference_feature], features_to_chose_from, DISTANCE_TYPE)
        verification_match_set[sequence_index].add_match(match)

################################################ PRINT RESULTS ##############################################################

set_nums_possible_correct_matches = np.array(set_nums_possible_correct_matches)
set_nums_possible_correct_matches.flatten()

set_repeatabilities = np.array(set_repeatabilities)
set_repeatabilities.flatten()

print(f"cm_total: mean {np.mean(set_nums_possible_correct_matches)} standard_deviation: {np.std(set_nums_possible_correct_matches)}")
print(f"rep_total: mean {np.mean(set_repeatabilities)} standard_deviation: {np.std(set_repeatabilities)}")
print()


print(f"total num matches: {sum(len(match_sequence) for match_sequence in matching_match_set)}")

total_possible_correct_matches = sum(
    num_correct_matches
    for num_correct_sequenceuence_matches in set_nums_possible_correct_matches
    for num_correct_matches in num_correct_sequenceuence_matches
)

print(f"num possible correct matches: {total_possible_correct_matches}")

total_correct_matches = sum(
    1 if match.is_correct else 0
    for match_sequence in matching_match_set
    for match in match_sequence
)

print(f"num correct matches: {total_correct_matches}")

# Results from matching
for match_rank_property in matching_approach.match_rank_properties:
    mAP = np.average([match_sequence.get_average_precision_score(match_rank_property) for match_sequence in matching_match_set])
    print(f"Matching {match_rank_property.name} mAP: {mAP}")

# Results from verification
for match_rank_property in matching_approach.match_rank_properties:
    mAP = np.average([match_sequence.get_average_precision_score(match_rank_property) for match_sequence in verification_match_set])
    print(f"Verification {match_rank_property.name} mAP: {mAP}")

# Results from retrieval
for match_rank_property in matching_approach.match_rank_properties:
    mAP = np.average([match_sequence.get_average_precision_score(match_rank_property) for match_sequence in retrieval_match_set])
    print(f"Retrieval {match_rank_property.name} mAP: {mAP}")