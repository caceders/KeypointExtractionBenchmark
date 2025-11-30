from benchmark.debug import display_feature_for_sequence, display_feature_in_image
from benchmark.feature import Feature
from benchmark.feature_extractor import FeatureExtractor
from benchmark.image_feature_set import ImageFeatureSet, ImageFeatureSequence
from benchmark.matching import Match, MatchSet, MatchingApproach, MatchRankProperty,homographic_optimal_matching, greedy_maximum_bipartite_matching
from benchmark.utils import load_HPSequences
from tqdm import tqdm
import cv2
import math
import numpy as np
import random
import warnings

DEBUG = "all" # all/matching/retrival/verification

###################################### SETUP TESTBENCH HERE #################################################################

## Set constants and configs.
MAX_FEATURES = 20
RELATIVE_SCALE_DIFFERENCE_THRESHOLD = 100
ANGLE_THRESHOLD = 180
DISTANCE_THRESHOLD = 10
DISTANCE_TYPE = cv2.NORM_HAMMING # cv2.NORM_L2 | cv2.NORM_HAMMING
VERIFICATION_CORRECT_TO_RANDOM_RATIO = 5
RETRIEVAL_CORRECT_TO_RANDOM_RATIO = 4000

## Load dataset.
dataset_image_sequences, dataset_homography_sequence = load_HPSequences(r"hpatches-sequences-release")

## Setup feature extractor.
ORB = cv2.ORB_create(nfeatures = MAX_FEATURES * 2)

feature_extractor = FeatureExtractor.from_opencv(ORB.detect, ORB.compute, True, 31, 31)

## Setup matching approach
distance_match_rank_property = MatchRankProperty("distance", False)
average_response_match_rank_property = MatchRankProperty("average_response", True)
average_ratio_match_rank_property = MatchRankProperty("average_ratio", False)
matching_properties = [distance_match_rank_property, average_response_match_rank_property, average_ratio_match_rank_property]

matching_approach = MatchingApproach(greedy_maximum_bipartite_matching, matching_properties)

#############################################################################################################################
warnings.filterwarnings("once", category=UserWarning)


num_sequences = len(dataset_image_sequences)
num_related_images = len(dataset_image_sequences[0]) - 1
image_feature_set = ImageFeatureSet(num_sequences, num_related_images)

time_per_image = []
## Speed test
for sequence_index, image_sequence in enumerate(tqdm(dataset_image_sequences, leave=False, desc="Calculating speed")):
    for image_index, image in enumerate(image_sequence):
        time = feature_extractor.get_extraction_time_on_image(image)
        time_per_image.append(time)

speed = np.average(time_per_image)

## Find features in all images.
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
        
        image_feature_set[sequence_index][image_index].extend(features)

        keypoints = [feature.keypoint for feature in features]
        descriptions = [feature.description for feature in features]


## Calculate valid matches.
for sequence_index, image_feature_sequence in enumerate(tqdm(image_feature_set, leave=False, desc="Calculating and ranking all valid matches")):
    for reference_feature in tqdm(image_feature_sequence.reference_image, leave=False):
        for related_image_index, related_image in enumerate(image_feature_sequence.related_images):
            img_size = len(dataset_image_sequences[sequence_index][related_image_index+1])
            for related_feature in related_image:

                homography = dataset_homography_sequence[sequence_index][related_image_index]
                transformed_related_feature_pt = related_feature.get_pt_after_homography_transform(homography)
                distance = math.dist(reference_feature.pt, transformed_related_feature_pt)
                if distance > (DISTANCE_THRESHOLD * feature_extractor.get_description_image_scale_factor(img_size)):
                    continue

                reference_feature_size = reference_feature.keypoint.size
                related_feature_size = related_feature.get_size_after_homography_transform(homography)
                biggest_keypoint = max(reference_feature_size, related_feature_size)
                smallest_keypoint = min(reference_feature_size, related_feature_size)
                relative_scale_differnce = abs(1 - biggest_keypoint/smallest_keypoint)
                if relative_scale_differnce > RELATIVE_SCALE_DIFFERENCE_THRESHOLD:
                    continue
                
                reference_feature_angle = reference_feature.keypoint.angle
                related_feature_angle = related_feature.get_angle_after_homography(homography)
                # Calculate the circular angle distance
                angle_difference = abs((reference_feature_angle - related_feature_angle + 180) % 360 - 180)
                if angle_difference > ANGLE_THRESHOLD:
                    continue

                reference_feature.store_valid_match_for_image(related_image_index, related_feature, distance)
                related_feature.store_valid_match_for_image(0, reference_feature, distance)



##############################################################################
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
##############################################################################



## Calculate repeatability and number of possible matches.
set_nums_possible_correct_matches= []
set_repeatabilities = []
for sequence_index, image_feature_sequence in enumerate(tqdm(image_feature_set, leave=False, desc="Calculating optimal matching results")):

    nums_possible_correct_matches = []
    repeatabilities = []
    
    for related_image_index, related_image in enumerate(image_feature_sequence.related_images):

        img_size = len(dataset_image_sequences[sequence_index][related_image_index+1])
        
        reference_features = image_feature_sequence.reference_image.copy()
        related_features = image_feature_sequence.related_image(related_image_index).copy()

        homography = dataset_homography_sequence[sequence_index][related_image_index]

        matches = homographic_optimal_matching(reference_features, related_features, homography) 

        # Check which matches were correct
        num_possible_correct_matches = 0
        for match in matches:
            if match.score < DISTANCE_THRESHOLD * feature_extractor.get_description_image_scale_factor(img_size):
                
                feature1 = match.feature1
                feature2 = match.feature2
                
                homography = dataset_homography_sequence[sequence_index][related_image_index]
                transformed_feature_2_pt = feature2.get_pt_after_homography_transform(homography)
                distance = math.dist(feature1.pt, transformed_feature_2_pt)
                if distance > (DISTANCE_THRESHOLD * feature_extractor.get_description_image_scale_factor(img_size)):
                    continue

                feature1_size = feature1.keypoint.size
                feature2_size = feature2.get_size_after_homography_transform(homography)
                biggest_keypoint = max(feature1_size, feature2_size)
                smallest_keypoint = min(feature1_size, feature2_size)
                relative_scale_differnce = abs(1 - biggest_keypoint/smallest_keypoint)
                if relative_scale_differnce > RELATIVE_SCALE_DIFFERENCE_THRESHOLD:
                    continue
                
                feature1_angle = feature1.keypoint.angle
                feature2_angle = feature2.get_angle_after_homography(homography)
                # Calculate the circular angle distance
                angle_difference = abs((feature1_angle - feature2_angle + 180) % 360 - 180)
                if angle_difference > ANGLE_THRESHOLD:
                    continue

                num_possible_correct_matches += 1

        repeatability = num_possible_correct_matches/len(reference_features)

        nums_possible_correct_matches.append(num_possible_correct_matches)
        repeatabilities.append(repeatability)

    set_nums_possible_correct_matches.append(nums_possible_correct_matches)
    set_repeatabilities.append(repeatabilities)



## Calculate matching results.
matching_match_sets: list[MatchSet] = []
if DEBUG == "all" or DEBUG == "matching":
    for sequence_index, image_feature_sequence in enumerate(tqdm(image_feature_set, leave=False, desc="Calculating matching results")):
        matching_match_set = MatchSet()
        matching_match_sets.append(matching_match_set)
        for related_image_index, related_image in enumerate(image_feature_sequence.related_images):

            # Reference and related features.
            reference_features = image_feature_sequence.reference_image.copy()
            related_features = image_feature_sequence.related_image(related_image_index).copy()

            matches = matching_approach.matching_callback(reference_features, related_features, DISTANCE_TYPE)
            matching_match_set.add_match(matches)




## Calculate verification results.
verification_match_sets: list [MatchSet] = []
if DEBUG == "all" or DEBUG == "verification":
    for sequence_index, image_feature_sequence in enumerate(tqdm(image_feature_set, leave=False, desc="Calculating verification results")):
        verification_match_set = MatchSet()
        verification_match_sets.append(verification_match_set)
        reference_features = image_feature_sequence.reference_image.copy()
        for reference_feature in reference_features:

            
            # Find related images with equivalent feature
            related_images_to_use = []

            for image_index in range(len(image_feature_sequence.related_images)):
                if isinstance(reference_feature.get_valid_matches_for_image(image_index), dict):
                    related_images_to_use.append(image_index)
            
            num_random_images = len(related_images_to_use) * VERIFICATION_CORRECT_TO_RANDOM_RATIO

            # Match for all relevant related images
            for related_image_index in range(len(related_images_to_use)):

                # Reference and related features.
                related_features = image_feature_sequence.related_image(related_image_index).copy()

                matches = matching_approach.matching_callback([reference_feature], related_features, DISTANCE_TYPE)
                verification_match_set.add_match(matches)
            
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
                random_image_features = image_feature_set[random_sequence_index][image_index].copy()
                matches = matching_approach.matching_callback([reference_feature], random_image_features, DISTANCE_TYPE)
                verification_match_set.add_match(matches)



## Calculate retrieval results.
retrieval_match_sets : list[MatchSet] = []

if DEBUG == "all" or DEBUG == "retrieval":
    for sequence_index, image_feature_sequence in enumerate(tqdm(image_feature_set, leave=False, desc="Calculating retrieval results")):
        
        retrieval_match_set = MatchSet()
        retrieval_match_sets.append(retrieval_match_set)

        reference_features = image_feature_sequence.reference_image.copy()
        this_image = image_feature_sequence.reference_image
        all_features_except_this_image = [feature 
                                            for choice_image_feature_sequence in image_feature_set
                                            for choice_image_features in choice_image_feature_sequence
                                            if choice_image_features is not this_image
                                            for feature in choice_image_features.copy()]
        
        for reference_feature in tqdm(reference_features, leave=False):
            
            correct_features = reference_feature.get_all_valid_matches()
            num_random_features = len(correct_features) * RETRIEVAL_CORRECT_TO_RANDOM_RATIO
            
            # Pick random features
            
            if num_random_features > len(all_features_except_this_image):
                warnings.warn(f"Not enough features to fully calculate retrieval.", UserWarning)
                num_random_features = len(all_features_except_this_image)

            chosen_random_features = random.sample(all_features_except_this_image, num_random_features)
            features_to_chose_from = correct_features + chosen_random_features
            
            # Match
            match = matching_approach.matching_callback([reference_feature], features_to_chose_from, DISTANCE_TYPE)
            retrieval_match_set.add_match(match)

################################################ PRINT RESULTS ##############################################################

set_nums_possible_correct_matches = np.array(set_nums_possible_correct_matches)
set_nums_possible_correct_matches.flatten()

set_repeatabilities = np.array(set_repeatabilities)
set_repeatabilities.flatten()

print(f"speed: {speed}")

print(f"cm_total: mean {np.mean(set_nums_possible_correct_matches)} standard_deviation: {np.std(set_nums_possible_correct_matches)}")
print(f"rep_total: mean {np.mean(set_repeatabilities)} standard_deviation: {np.std(set_repeatabilities)}")
print()


print(f"total num matches: {sum(len(match_set) for match_set in matching_match_sets)}")

total_possible_correct_matches = sum(
    num_correct_matches
    for num_correct_sequence_matches in set_nums_possible_correct_matches
    for num_correct_matches in num_correct_sequence_matches
)

print(f"num possible correct matches: {total_possible_correct_matches}")

total_correct_matches = sum(
    1 if match.is_correct else 0
    for match_set in matching_match_sets
    for match in match_set
)

print(f"num correct matches: {total_correct_matches}")

# Results from matching
for match_rank_property in matching_approach.match_properties:
    mAP = np.average([match_set.get_average_precision_score(match_rank_property) for match_set in matching_match_sets])
    print(f"Matching {match_rank_property.name} mAP: {mAP}")

# Results from verification
for match_rank_property in matching_approach.match_properties:
    mAP = np.average([match_set.get_average_precision_score(match_rank_property) for match_set in verification_match_sets])
    print(f"Verification {match_rank_property.name} mAP: {mAP}")

# Results from retrieval
for match_rank_property in matching_approach.match_properties:
    mAP = np.average([match_set.get_average_precision_score(match_rank_property) for match_set in retrieval_match_sets])
    print(f"Retrieval {match_rank_property.name} mAP: {mAP}")