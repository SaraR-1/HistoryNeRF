import numpy as np
from pathlib import Path
import json
import warnings

from nerfstudio.utils.io import load_from_json

class NotImplementedWarning(UserWarning):
    pass

class Pose:
    """Class to handle the computation of pose related measurements"""

    @staticmethod
    def compute_angular_error(rotation1, rotation2):
        """Computes the angular error between two rotations"""
        # Relative rotation matrix between two matrices. First undo the rotation of R1 (transpose) and then apply the rotation of R2, giving the rotation from R1 to R2
        r_rel = rotation1.T @ rotation2
        # Trace of the relative rotation matrix (sum of diagonal elements)
        # In 3D, the trace of a rotation matrix that rotates a vector by an angle theta around a unit vector is 1 + 2*cos(theta)
        cos_theta = (np.trace(r_rel) - 1) / 2
        # Compute the angle of rotation
        theta = np.arccos(cos_theta.clip(-1, 1))
        # Convert the angle of rotation from radians to degrees
        return theta * 180 / np.pi 

    @staticmethod
    def report_error(errors, max_distance):
        """Reports the percentage of errors less than a max distance"""
        percentage = sum([1 for i in errors if i <= max_distance]) / len(errors) * 100
        return percentage

    @staticmethod
    def compute_l2_translation(translation1, translation2):
        """Computes the l2 norm between two translations and the scene scale"""
        l2_norm = np.linalg.norm(translation1 - translation2)
        gt_translation = translation1 if len(translation1) > len(translation2) else translation2
        scene_scale = np.linalg.norm(gt_translation)
        return l2_norm, scene_scale


class Transformation:
    """
    Class to handle transformations related operations
    
    This transformation does not work wheb there is only one camera model - how to handle that?
    """

    @staticmethod
    def estimate_transformation(points1, points2):
        """Estimates the transformation between two sets of points"""
        centroid1, centroid2 = np.mean(points1, axis=0), np.mean(points2, axis=0)
        covariance_matrix = (points2 - centroid2).T @ (points1 - centroid1)
        u_mat, _, vt_mat = np.linalg.svd(covariance_matrix)
        rotation_matrix = vt_mat @ u_mat.T
        translation_vector = np.mean(points2 - (points1 @ rotation_matrix), axis=0)
        return rotation_matrix, translation_vector

    @staticmethod
    def apply_transformation(rotation_matrix, translation_vector, points):
        """Applies a transformation to a set of points"""
        transformed_points = points @ rotation_matrix + translation_vector
        return transformed_points

def image_name_as_id(camera_json):
    """Converts image names to ids"""
    return {f["file_path"]: np.array(f["transform_matrix"][:-1]) for f in camera_json["frames"]}

def cameras_from_json(path):
    cameras_json = load_from_json(path)
    camera_pp = np.array([cameras_json["cx"], cameras_json["cy"]])
    cameras = image_name_as_id(cameras_json)

    return camera_pp, cameras

def compare_poses(
    camera_path1, 
    camera_path2, 
    angular_error_max_dist=15, 
    translation_error_max_dist=0.25,
    output_dir=""
    ):

    """Compares poses from two different reconstructions"""
    camera1_pp, cameras1 = cameras_from_json(camera_path1)
    camera2_pp, cameras2 = cameras_from_json(camera_path2)

    if (camera1_pp != camera2_pp).any():
        print(f"{camera1_pp}, {camera2_pp}")
        warnings.warn("Different camera models not supported yet. Transofrmation estimation not implemented.", NotImplementedWarning)
        # raise  NotImplementedError("Different camera models not supported yet. Transofrmation estimation not implemented.")
        # if len(cameras1) < len(cameras2):
        #     rotation, translation = Transformation.estimate_transformation(camera1_pp, camera2_pp)
        #     images1_byname = Transformation.apply_transformation(rotation, translation, images1_byname)
        # else:
        #     rotation, translation = Transformation.estimate_transformation(camera2_pp, camera1_pp)
        #     images2_byname = Transformation.apply_transformation(rotation, translation, images2_byname)

    common_cameras = list(set(cameras1.keys()).intersection(cameras2.keys()))
    print(f"Number of common cameras: {len(common_cameras)}")

    angular_errors, translation_errors, scenes_scale, translation_errors_rescale = [], [], [], []
    for camera in common_cameras:
        rotation1, translation1 = cameras1[camera][:3, :3], cameras1[camera][:3, 3]
        rotation2, translation2 = cameras2[camera][:3, :3], cameras2[camera][:3, 3]

        angular_error = Pose.compute_angular_error(rotation1, rotation2)
        translation_error, scene_scale = Pose.compute_l2_translation(translation1, translation2)

        angular_errors.append(angular_error)
        translation_errors.append(translation_error)
        translation_errors_rescale.append(translation_error / scene_scale)

    aggregate_angular_error_dict = {
        "mean": np.mean(angular_errors),
        "median": np.median(angular_errors),
        "std": np.std(angular_errors),
    }

    aggregate_translation_error_dict = {
        "mean": np.mean(translation_errors),
        "median": np.median(translation_errors),
        "std": np.std(translation_errors),
        "mean_rescale": np.mean(translation_errors_rescale),
        "median_rescale": np.median(translation_errors_rescale),
        "std_rescale": np.std(translation_errors_rescale),
    }

    # Save metrics vectors in a file 
    metrics_dict = {"Angular Error": aggregate_angular_error_dict, "Translation Error": aggregate_translation_error_dict}
    with open(output_dir / "colmap_metrics.json", "w") as f:
        json.dump(metrics_dict, f)


if __name__ == "__main__":
    # Arguments
    camera_path1 = Path("/workspace/data/bridge_of_sighs/data/train/transforms.json")
    camera_path2 = Path("/workspace/data/bridge_of_sighs/data/train/processed_data/transforms.json")
    output_dir = Path("/workspace/data/bridge_of_sighs")

    compare_poses(
        camera_path1=camera_path1, 
        camera_path2=camera_path2, 
        angular_error_max_dist=15, 
        translation_error_max_dist=0.25, 
        output_dir=output_dir)
    

# python3 historynerf/evaluation/nerf.py 