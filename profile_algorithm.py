from shi_tomasi_sift import ShiTomasiSift
import cv2
from tqdm import tqdm

gftt = cv2.GFTTDetector.create()
sift = cv2.SIFT.create()

algorithm = ShiTomasiSift(calculate_orientation_for_keypoints= True,
                              scale_pyramid_scaling_factor=1.3,
                              response_type="normal",
                              scale_pyramid_blur_sigma=-1,
                              orientation_calculation_gaussian_weight_std=50,
                              num_octaves_in_scale_pyramid=6,
                              descriptor_gaussian_weight_std=50,
                              derivation_operator="simple",
                              d_weight=0.7,
                              base_blur_sigma=0.7,
                              )


img = cv2.imread(r"shi_tomasi_sift\test.jpg", cv2.IMREAD_COLOR_BGR)
img = cv2.resize(img, dsize = (500, 500))
kps = algorithm.detect(img)
algorithm.compute(img, kps)