import cv2
import os
import numpy as np
from pathlib import Path
from typing import Protocol, List, Tuple, Any
from enum import Enum

class NormTypes(Enum):
    NORM_INF = cv2.NORM_INF
    NORM_L1 = cv2.NORM_L1
    NORM_L2 = cv2.NORM_L2
    NORM_L2SQR = cv2.NORM_L2SQR
    NORM_HAMMING = cv2.NORM_HAMMING
    NORM_HAMMING2 = cv2.NORM_HAMMING2
    NORM_TYPE_MASK = cv2.NORM_TYPE_MASK
    NORM_RELATIVE = cv2.NORM_RELATIVE
    NORM_MINMAX = cv2.NORM_MINMAX

# Define a protocol for keypoint detectors
class KeypointDetectorProtocol(Protocol):
    def detectAndCompute(self, image: np.ndarray, mask: Any) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        pass
        
class ImageAligner:
    def __init__(
        self, 
        image_directory: Path, 
        keypoint_detector: KeypointDetectorProtocol,
        matcher_distance: NormTypes,
        match_filter: float = 0.75):
        """Evaluate the pair-wise alignemt of all the images in directory

        Args:
            image_directory (Path): input directory containing all images to include in the evaluation
            keypoint_detector (cv2): detector used to extract features and descriptions
            matcher_distance (cv2.NormTypes): distance used to match keypoint
            match_filter (float): ratio to filter good matches
        """
        self.image_directory = image_directory
        self.images = self._load_images()
        
        # TODO: check cv2.SIFT_create() and cv2.ORB_create() 
        self.keypoint_detector = keypoint_detector
        # Using BFMatcher with default params and L2 norm (suitable for SIFT)
        # cv2.NORM_HAMMING with cv2.ORB_create(), L2 with sift
        self.bf = cv2.BFMatcher(matcher_distance.value, crossCheck = True)
        self.match_filter = match_filter
    
    def _load_images(self):    
        image_files = [f for f in self.image_directory.iterdir() if f.suffix in ['.jpg', '.jpeg', '.png']]
        return [cv2.imread(str(f)) for f in image_files]
    
    def compute_keypoint_detector(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.keypoint_detector.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def match_keypoints(self, desc1, desc2):
        matches = self.bf.match(desc1, desc2)

        # Filter matches based on distance
        good_matches = [m for m in matches if m.distance < self.match_filter]
        
        return good_matches
    
    def align_images(self):
        results = []

        for i in range(len(self.images)):
            for j in range(i+1, len(self.images)):
                img1 = self.images[i]
                img2 = self.images[j]
                
                # Extract SIFT keypoints and descriptors
                kp1, desc1 = self.compute_keypoint_detector(img1)
                kp2, desc2 = self.compute_keypoint_detector(img2)
                
                # Match keypoints
                matches = self.match_keypoints(desc1, desc2)
                
                # findHomography requires at least 4 pairs of matching points to compute a homography matrix
                if len(matches) >=4 :
                    # Extract matched keypoints
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    
                    # Estimate homography
                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    
                    results.append((i, j, H))
                else:
                    results.append((i, j, None))
        
        return results
    
    def visualize_keypoints(self, image_index: int) -> np.ndarray:
        """
        Visualize keypoints on the specified image.
        
        :param image_index: Index of the image in the loaded image list.
        :return: Image with keypoints drawn.
        """
        image = self.images[image_index]
        keypoints, _ = self.compute_keypoints(image)
        return cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    def visualize_alignment(self, image1_index: int, image2_index: int, H: np.ndarray) -> np.ndarray:
        """
        Visualize alignment of the two images using the provided homography matrix.
        
        :param image1_index: Index of the first image.
        :param image2_index: Index of the second image.
        :param H: Homography matrix to align image1 to image2.
        :return: Aligned image.
        """
        image1 = self.images[image1_index]
        image2 = self.images[image2_index]
        
        # Warp the first image into the coordinate space of the second image
        aligned_img = cv2.warpPerspective(image1, H, (image2.shape[1] + image1.shape[1], image2.shape[0]))
        
        # Place the second image on the aligned image
        aligned_img[0:image2.shape[0], 0:image2.shape[1]] = image2
        
        return aligned_img

    def _get_img_idx(self, img_name: str) -> int:
        try:
            return self.images.index(img_name)
        except ValueError:
            raise NameError(f"Image '{img_name}' not found in the loaded images.")
        
    @staticmethod
    def normalized_overlap(image1: np.ndarray, image2: np.ndarray) -> float:
        overlap = np.where((image1 > 0) & (image2 > 0))
        return len(overlap[0]) / (image2.shape[0] * image2.shape[1])
    
    @staticmethod
    def compute_ncc(image1: np.ndarray, image2: np.ndarray) -> float:
        # Ensure both images are of same shape
        if image1.shape != image2.shape:
            raise ValueError("Both images must have the same shape.")
            
        mean1 = np.mean(image1)
        mean2 = np.mean(image2)
        
        numerator = np.sum((image1 - mean1) * (image2 - mean2))
        denominator = np.sqrt(np.sum((image1 - mean1)**2) * np.sum((image2 - mean2)**2))
        
        return numerator / denominator

    @staticmethod
    def compute_mse(image1: np.ndarray, image2: np.ndarray) -> float:
        if image1.shape != image2.shape:
            raise ValueError("Both images must have the same shape.")
        return np.mean((image1 - image2) ** 2)
    
    @staticmethod
    def compute_ssd(image1: np.ndarray, image2: np.ndarray) -> float:
        if image1.shape != image2.shape:
            raise ValueError("Both images must have the same shape.")
        return np.sum((image1 - image2) ** 2)
    

images_path = Path("/home/sara/every6frames/images")
aligner = ImageAligner(images_path, keypoint_detector=cv2.SIFT_create(),  matcher_distance=NormTypes.NORM_L2, match_filter=0.4)
aligned_images = aligner.align_images()

# Visualize keypoints for the first image
cv2.imshow('Keypoints', aligner.visualize_keypoints(0))
cv2.waitKey(0)

# Visualize alignment for the first and second images
aligned_images = aligner.align_images()  # Assuming align_images is still returning a list of (i, j, H) tuples
_, _, H = aligned_images[0]  # Get the homography for the first pair
cv2.imshow('Aligned Image', aligner.visualize_alignment(0, 1, H))
cv2.waitKey(0)

breakpoint()

# aligned_img1 = cv2.warpPerspective(image1, H, (image2.shape[1] + image1.shape[1], image2.shape[0]))
# overlap_img1 = aligned_img1[0:image2.shape[0], 0:image2.shape[1]]
# overlap_img2 = image2

# ncc_val = compute_ncc(overlap_img1, overlap_img2)
# mse_val = compute_mse(overlap_img1, overlap_img2)
# ssd_val = compute_ssd(overlap_img1, overlap_img2)
# overlap_val = normalized_overlap(overlap_img1, overlap_img2)

