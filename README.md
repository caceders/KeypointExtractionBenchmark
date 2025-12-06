TODO: Wife stealing problem, how to do ratio when using second choice, JUST REMOVE MASK???

NOW:
- Create tests for:
    - calculate_valid_matches
    - Matching
        - Hamming

- Update documentation

LATER:
- Fix


IF TIME:
- Make hashing actually work with the features in dictionaries.
- Create conftest for more ordered fixtures in pytest
- Create tests for    
    - Check for type in lists within lists?
    - Features (edge cases)
    - Feature extractor (edge cases)
    - Image feature sequence (edge cases)
    - Image feature set (edge cases)
    - Match set (edge cases)
    - speed_test (edge cases)
    - find_all_features_for_dataset (edge cases)
    - calculate_matching_evaluation (edge cases)
    - calculate_verification_results (edge cases)
    - calculate_retrieval_results (edge cases)


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
        - It is not "mock" - find a better name
        - Does the general cases explain themselves or need documentation?
        - Magic numbers?

    - Check main source code
        - When mentioning specific arguments in the error messages use the actual names "x_y_z", not "x y z"