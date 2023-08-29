import cv2
import os
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Protocol, List, Tuple, Any
from tqdm import tqdm
from enum import Enum
import json

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
    
KEYPOINT_DETECTOR_MAPPING = {
    "orb": cv2.ORB_create(),
    "sift": cv2.SIFT_create()
}

NORM_MAPPING = {
    "NORM_INF": cv2.NORM_INF,
    "NORM_L1": cv2.NORM_L1,
    "NORM_L2": cv2.NORM_L2,
    "NORM_L2SQR": cv2.NORM_L2SQR,
    "NORM_HAMMING": cv2.NORM_HAMMING,
    "NORM_HAMMING2": cv2.NORM_HAMMING2,
    "NORM_TYPE_MASK": cv2.NORM_TYPE_MASK,
    "NORM_RELATIVE": cv2.NORM_RELATIVE,
    "NORM_MINMAX": cv2.NORM_MINMAX,
    
}

def get_keypoint_detector(detector_str: str):
    try:
        return KEYPOINT_DETECTOR_MAPPING[detector_str]
    except ValueError:
        raise NameError(f"keypoint Detector {detector_str} not supported.")

def get_norm_type(norm_str: str):
    try:
        return NORM_MAPPING[norm_str]
    except ValueError:
        raise NameError(f"Norm {norm_str} not supported.")
     
class ImageAligner:
    def __init__(
        self, 
        image_directory: Path, 
        keypoint_detector: str,
        matcher_distance: str,
        match_filter: float, 
        matched_keypoints_threshold: int):
        """Evaluate the pair-wise alignemt of all the images in directory

        Args:
            image_directory (Path): input directory containing all images to include in the evaluation
            keypoint_detector (KeypointDetectorProtocol): detector used to extract features and descriptions. E.g. cv2.SIFT_create() and cv2.ORB_create() 
            matcher_distance (NormTypes): distance used to match keypoint
            match_filter (float): ratio to filter good matches
            matched_keypoints_threshold (int): threshold for the number of good matches required to consider the homography estimation as reliable
        """
        self.image_directory = image_directory
        self.image_files, self.images, self.image_keypoints, self.image_descriptors = self._load_images_and_compute_keypoints(get_keypoint_detector(keypoint_detector))
        _, self.coloured_images = self._load_images(cv2.IMREAD_COLOR)
        
        # cv2.NORM_HAMMING with cv2.ORB_create(), L2 with sift
        # self.bf = cv2.BFMatcher(matcher_distance.value, crossCheck = True)
        
        # Using FlannBasedMatcher for faster matching
        matcher_distance = get_norm_type(matcher_distance)
        if matcher_distance in [NormTypes.NORM_L1, NormTypes.NORM_L2]:
            self.flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})
        else:
            index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=50)
            self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        self.match_filter = match_filter
        self.matched_keypoints_threshold = matched_keypoints_threshold
    
    def _load_images(self, colour_scale=cv2.IMREAD_GRAYSCALE):    
        image_files = [f for f in self.image_directory.iterdir() if f.suffix in ['.jpg', '.jpeg', '.png']]
        return image_files, [cv2.imread(str(f), colour_scale) for f in image_files]  # Convert to grayscale directly
    
    def _load_images_and_compute_keypoints(self, keypoint_detector):    
        image_files, images = self._load_images()

        keypoints_list = []
        descriptors_list = []
        for img in tqdm(images, desc="Computing keypoints"):
            keypoints, descriptors = keypoint_detector.detectAndCompute(img, None)
            keypoints_list.append(keypoints)
            descriptors_list.append(descriptors)
            
        return image_files, images, keypoints_list, descriptors_list

    def match_keypoints(self, desc1, desc2):
        matches = self.flann.knnMatch(desc1, desc2, k=2)
        
        # Filter matches using Lowe's ratio test
        # breakpoint()
        # good_matches = [m for m, n in matches if m.distance < self.match_filter * n.distance]
        
        good_matches = [m for match in matches if len(match) == 2 for m, n in [match] if m.distance < self.match_filter * n.distance]
    
        
        return good_matches
    
    def align_images(self):
        results = []
        
        total_iterations = (len(self.images) * (len(self.images) - 1)) // 2
        pbar = tqdm(total=total_iterations, desc="Aligning images")
        
        for i in range(len(self.images)):
            for j in range(i+1, len(self.images)):
                desc1 = self.image_descriptors[i]
                desc2 = self.image_descriptors[j]
                
                # Match keypoints
                matches = self.match_keypoints(desc1, desc2)
                
                # findHomography requires at least 4 pairs of matching points to compute a homography matrix
                if len(matches) >= self.matched_keypoints_threshold:
                    # Extract matched keypoints
                    src_pts = np.float32([self.image_keypoints[i][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([self.image_keypoints[j][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    
                    # Estimate homography
                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    
                    results.append((i, j, H))
                else:
                    results.append((i, j, None))
                
                # Update the progress bar after each inner loop iteration
                pbar.update(1)
        
        pbar.close()
        
        return results
    
    def visualize_keypoints(self, image_index: int) -> np.ndarray:
        """
        Visualize keypoints on the specified image.
        
        :param image_index: Index of the image in the loaded image list.
        :return: Image with keypoints drawn.
        """
        image = self.coloured_images[image_index]
        keypoints = self.image_keypoints[image_index]
        return cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
   
    def visualize_keypoint_matches(self, image1_index: int, image2_index: int):
        """
        Visualizes the keypoint matches between two images.
        
        :param image1_index: Index of the first image.
        :param image2_index: Index of the second image.
        :return: Image with drawn keypoint matches.
        """
        img1 = self.images[image1_index]
        img2 = self.images[image2_index]
        kp1 = self.image_keypoints[image1_index]
        kp2 = self.image_keypoints[image2_index]
        desc1 = self.image_descriptors[image1_index]
        desc2 = self.image_descriptors[image2_index]
        
        matches = self.match_keypoints(desc1, desc2)
        
        return cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    def visualize_alignment(self, 
                            image1_index: int, 
                            image2_index: int, 
                            H: np.ndarray) -> np.ndarray:
        """
        Visualize alignment of the two images using the provided homography matrix.
        
        :param image1_index: Index of the first image.
        :param image2_index: Index of the second image.
        :param H: Homography matrix to align image2 to image1.
        :return: Aligned image.
        """
        image1 = self.coloured_images[image1_index]
        image2 = self.coloured_images[image2_index]
        
        # Warp the second image into the coordinate space of the first image
        warped_image2 = cv2.warpPerspective(image2, np.linalg.inv(H), (image1.shape[1] + image2.shape[1], image1.shape[0]))
        
        # Create a result image that can fit both the original first image and the warped second image
        result = np.zeros_like(warped_image2)
        
        # Place the original first image into this result image
        result[0:image1.shape[0], 0:image1.shape[1]] = image1
        
        # Overlay the warped second image onto the result
        for i in range(warped_image2.shape[0]):
            for j in range(warped_image2.shape[1]):
                if np.all(warped_image2[i, j] == 0):  # if it's all black in the warped image
                    continue
                result[i, j] = warped_image2[i, j]
        
        # Calculate the transformed corners of the second image for visualization
        h, w = image2.shape[:2]
        corners_image2 = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners_image2, np.linalg.inv(H))
        
        # Draw the transformed corners on the result
        cv2.polylines(result, [np.int32(transformed_corners)], isClosed=True, color=(255, 0, 0), thickness=3)
        
        return result

    def _get_img_idx(self, img_name: str) -> int:
        try:
            return self.image_files.index(img_name)
        except ValueError:
            raise NameError(f"Image '{img_name}' not found in the loaded images.")
        
    def get_img_name(self, img_idx: str) -> int:
        try:
            return self.image_files[img_idx].stem
        except ValueError:
            raise NameError(f"Image '{img_idx}' not found in the loaded images.")
        
    @staticmethod
    def normalized_overlap(image1: np.ndarray, image2: np.ndarray) -> float:
        overlap = np.where((image1 > 0) & (image2 > 0))
        total_area = image1.size + image2.size - len(overlap[0])  # Total combined area minus overlapping area
        return len(overlap[0]) / total_area
    
    @staticmethod
    def refined_normalized_overlap(image1: np.ndarray, image2: np.ndarray) -> float:
        """Calculate the normalized overlap between two images, considering pixel intensities."""
        # Ensure both images are of same shape
        if image1.shape != image2.shape:
            raise ValueError("Both images must have the same shape.")
        
        overlap = np.where((image1 > 0) & (image2 > 0))
        overlap_intensity = np.sum(image1[overlap] * image2[overlap])
        
        total_intensity = np.sum(image1) + np.sum(image2) - overlap_intensity  # Total combined intensity minus overlapping intensity
        return overlap_intensity / total_intensity

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

def save_alignment_scores_to_csv(alignment_scores, output_directory):
    # Convert the dictionary of lists into a DataFrame
    df_list = []
    
    for metric, scores in alignment_scores.items():
        temp_df = pd.DataFrame(scores, columns=["Image1 Index", "Image2 Index", f"{metric} Score"])
        
        if df_list:
            # If there are DataFrames in df_list, merge the new DataFrame with the existing ones on Image1 Index and Image2 Index
            df_list[0] = pd.merge(df_list[0], temp_df, on=["Image1 Index", "Image2 Index"])
        else:
            df_list.append(temp_df)
    
    # Save the DataFrame to a CSV file
    df_list[0].to_csv(output_directory / "alignment_scores.csv", index=False)
    

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_alignment_scores_to_json(alignment_scores, output_directory):
    # Convert the dictionary of lists into a structured dictionary for JSON
    structured_scores = {}
    for metric, scores in alignment_scores.items():
        structured_scores[metric] = [{"Image1 Index": i, "Image2 Index": j, "Score": score} for i, j, score in scores]
    
    # Save the structured dictionary to a JSON file
    with open(output_directory / "alignment_scores.json", 'w') as json_file:
        json.dump(structured_scores, json_file, indent=4, cls=NumpyEncoder)

def evaluate_and_visualize_alignment(
    image_directory: Path, 
    output_directory: Path,
    keypoint_detector: KeypointDetectorProtocol, 
    matcher_distance: NormTypes, 
    match_filter: float, 
    matched_keypoints_threshold: int):
    """
    Sample usage of the ImageAligner class to:
    1) Compute pair-wise alignment
    2) Quantify the pair-wise alignment
    3) Visualize the alignment for the worst, best, and an average pair

    Args:
        image_directory (Path): _description_
        output_directory (Path): _description_
        keypoint_detector (KeypointDetectorProtocol): _description_
        matcher_distance (NormTypes): _description_
        match_filter (float): _description_
        matched_keypoints_threshold (int): _description_
    """
    # Initialize the ImageAligner class
    aligner = ImageAligner(image_directory, keypoint_detector, matcher_distance, match_filter, matched_keypoints_threshold)
    
    # 1) Compute pair-wise alignment
    alignments = aligner.align_images()
    
    # 2) Quantify the pair-wise alignment using the normalized_overlap metric as an example
    # Metrics we want to compute for each pair
    metrics = {
        "normalized_overlap": ImageAligner.normalized_overlap,
        "ssd": ImageAligner.compute_ssd,
        "mse": ImageAligner.compute_mse,
        "ncc": ImageAligner.compute_ncc,
        "refined_normalized_overlap": ImageAligner.refined_normalized_overlap        
    }
    
    alignment_scores = {metric: [] for metric in metrics}
    for (i, j, H) in tqdm(alignments, desc="Save Pair-Wise Alignment"):
        if H is not None:
            aligned_img1 = cv2.warpPerspective(aligner.images[i], H, (aligner.images[j].shape[1] + aligner.images[i].shape[1], aligner.images[j].shape[0]))
            overlap_img1 = aligned_img1[0:aligner.images[j].shape[0], 0:aligner.images[j].shape[1]]
            overlap_img2 = aligner.images[j]
            
            aligned_visualization = aligner.visualize_alignment(i, j, H)
            keypoint_matches = aligner.visualize_keypoint_matches(i, j)
        
            # Save the visualized alignment
            output_filename = f"alignment_{aligner.get_img_name(i)}_to_{aligner.get_img_name(j)}.png"
            cv2.imwrite(str(output_directory / output_filename), aligned_visualization)

            output_filename = f"keypoint_matches_{aligner.get_img_name(i)}_to_{aligner.get_img_name(j)}.png"
            cv2.imwrite(str(output_directory / output_filename), keypoint_matches)
            
            for metric, func in metrics.items():
                score = func(overlap_img1, overlap_img2)
                alignment_scores[metric].append((i, j, score))
        else:
            for metric in metrics:
                alignment_scores[metric].append((i, j, float('inf') if metric != "ncc" else -float('inf')))     

    save_alignment_scores_to_csv(alignment_scores, output_directory)
    save_alignment_scores_to_json(alignment_scores, output_directory)
