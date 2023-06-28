import numpy as np
from pathlib import Path
import pycolmap

class Pose:
    """Class to handle the computation of pose related measurements"""

    @staticmethod
    def compute_angular_error(rotation1, rotation2):
        """Computes the angular error between two rotations"""
        r_rel = rotation1.T @ rotation2
        cos_theta = (np.trace(r_rel) - 1) / 2
        theta = np.arccos(cos_theta.clip(-1, 1))
        return theta * 180 / np.pi 

    @staticmethod
    def report_error(errors, max_distance):
        """Reports the percentage of errors less than a max distance"""
        percentage = sum([1 for i in errors if i < max_distance]) / len(errors) * 100
        return percentage

    @staticmethod
    def compute_l2_translation(translation1, translation2):
        """Computes the l2 norm between two translations and the scene scale"""
        l2_norm = np.linalg.norm(translation1 - translation2)
        gt_translation = translation1 if len(translation1) > len(translation2) else translation2
        scene_scale = np.linalg.norm(gt_translation)
        return l2_norm, scene_scale

class Images:
    """Class to handle image related operations"""

    @staticmethod
    def image_name_as_id(images):
        """Converts image names to ids"""
        return {i.name:i for i in images.values()}

class Transformation:
    """Class to handle transformations related operations"""

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

def compare_poses(path1, path2, angular_error_max_dist=15, translation_error_max_dist=0.25):
    """Compares poses from two different reconstructions"""
    path1, path2 = Path(path1), Path(path2)
    if not path1.is_dir() or not path2.is_dir():
        print("One or both paths do not exist")
        return None, None

    rec1, rec2 = pycolmap.Reconstruction(path1), pycolmap.Reconstruction(path2)
    images1_byname, images2_byname = Images.image_name_as_id(rec1.images), Images.image_name_as_id(rec2.images)
    images_name_comm = list(set(images1_byname.keys()).intersection(images2_byname.keys()))

    camera1_pp = [cameras1[images1_byname[i].camera_id].pp for i in images_name_comm]
    camera2_pp = [cameras2[images2_byname[i].camera_id].pp for i in images_name_comm]

    if camera1_pp != camera2_pp:
        if len(cameras1) < len(cameras2):
            rotation, translation = Transformation.estimate_transformation(camera1_pp, camera2_pp)
            images1_byname = Transformation.apply_transformation(rotation, translation, images1_byname)
        else:
            rotation, translation = Transformation.estimate_transformation(camera2_pp, camera1_pp)
            images2_byname = Transformation.apply_transformation(rotation, translation, images2_byname)

    angular_errors, l2_translations, scene_scale, l2_translations_rescale = [], [], [], []
    for i in images_name_comm:
        angular_errors.append(Pose.compute_angular_error(images1_byname[i].rotmat(), images2_byname[i].rotmat()))
        l2_translations_i, scene_scale_i = Pose.compute_l2_translation(images1_byname[i].tvec, images2_byname[i].tvec)
        l2_translations.append(l2_translations_i)
        scene_scale.append(scene_scale_i)
        l2_translations_rescale.append(l2_translations_i / scene_scale_i)

    aggregate_angular_error = Pose.report_error(angular_errors, angular_error_max_dist)
    aggregate_l2_translation = Pose.report_error(l2_translations, translation_error_max_dist)

    aggregate_angular_error_dict = {
        "percentage_max_dist": aggregate_angular_error, 
        "mean": np.mean(angular_errors),
        "median": np.median(angular_errors),
        "std": np.std(angular_errors),
    }

    aggregate_translation_error_dict = {
        "percentage_max_dist": aggregate_l2_translation, 
        "mean": np.mean(l2_translations),
        "median": np.median(l2_translations),
        "std": np.std(l2_translations),
        "mean_rescale": np.mean(l2_translations_rescale),
        "median_rescale": np.median(l2_translations_rescale),
        "std_rescale": np.std(l2_translations_rescale),
    }

    return aggregate_angular_error_dict, aggregate_translation_error_dict
