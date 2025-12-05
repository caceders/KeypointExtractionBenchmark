TODO: Wife stealing problem, how to do ratio when using second choice, JUST REMOVE MASK???

NOW:
- Remove the angles ONLY SMOOTHPEOPLE HEREEE!!!! >:c
- Create tests for:
    - speed_test
    - find_all_features_for_dataset
    - calculate_valid_matches
    - calculate_numbers_of_possible_correct_matches_and_repeatability
    - calculate_matching_results
    - calculate_verification_results
    - calculate_retrieval_results


    - Matching
        - Hamming <- Carl Anders jobber her

- Update documentation

LATER:
- Calculate homographic distance only once and use it in EITHER the calculations in main or in matching.

IF TIME:
- Make hashing actually work with the features in dictionaries.
- Create tests for    
    - Features (edge cases)
    - Feature extractor (edge cases)
    - Image feature sequence (edge cases)
    - Image feature set (edge cases)
    - Match set (edge cases)


CHECKLIST AFTER TESTS:
    - Does everything have:
        - Test for invalid input
        - Test that valid input pass
        - Test for general cases
        - Same formatting
        - Return typehints for fixtures and so on
        - Documentation strings
        - Rename all "All but x" to "Bad argument x"
        - Rename all indexed things from "x_1" to "x1"
        - When mentioning specific arguments in the error messages use the actual names "x_y_z", not "x y z"