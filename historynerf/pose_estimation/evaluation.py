import numpy as np
from pathlib import Path
import pycolmap

def compute_angular_error(rotation1, rotation2):
    '''
    Absolute Angle Difference between two rotation matrices. Code based on RelPose code.
    Args:
        rotation1: 3x3 rotation matrix
        rotation2: 3x3 rotation matrix
    Returns:
        angular_error: angle difference in degrees
    '''
    # Relative rotation matrix between two matrices. First undo the rotation of R1 (transpose) and then apply the rotation of R2, giving the rotation from R1 to R2
    R_rel = rotation1.T @ rotation2
    # Trace of the relative rotation matrix (sum of diagonal elements)
    tr = np.trace(R_rel)
    # In 3D, the trace of a rotation matrix that rotates a vector by an angle theta around a unit vector is 1 + 2*cos(theta)
    cos_theta = (tr - 1) / 2
    # Compute the angle of rotation
    theta = np.arccos(cos_theta.clip(-1, 1))
    # Convert the angle of rotation from radians to degrees
    return theta * 180 / np.pi 

def report_angular_error(angular_error, max_distance=15):
    '''
    Report the percentage of images with angular error less than max_distance
    Args:
        angular_error: list of angular errors
        max_distance: max distance to consider an image as correctly estimated
    Returns:
        percentage: percentage of images with angular error less than max_distance
    '''
    return sum([1 for i in angular_error if i < max_distance]) / len(angular_error) * 100

def report_l2_translation(l2_translation, scene_scale, max_distance=0.25):
    '''
    Report the percentage of images with L2 norm of the translation vector less than max_distance
    Args:
        l2_translation: list of L2 norm of the translation vectors
        max_distance: max distance to consider an image as correctly estimated
    Returns:
        percentage: percentage of images with L2 norm of the translation vector within max_distance of the scale of the scene
    '''
    return sum([1 for diff, scale in zip(l2_translation, scene_scale) if diff < max_distance*scale]) / len(l2_translation) * 100

def compute_l2_translation(translation1, translation2):
    '''
    L2 norm between two translation vectors
    Args:
        translation1: 3x1 translation vector
        translation2: 3x1 translation vector
    Returns:
        l2_norm: L2 norm between the two translation vectors
        scene_scale: scale of the scene, l2 norm of the ground truth translation
        '''
    # Compute the scale of the scene as the norm of the ground truth translation
    l2_norm = np.linalg.norm(translation1 - translation2)
    gt_translation = translation1 if len(translation1) > len(translation2) else translation2
    scene_scale = np.linalg.norm(gt_translation)
    return l2_norm, scene_scale

def image_name_asid(images):
    '''
    Use the image name as key for the images dictionary
    '''
    return {i.name:i for i in images.values()}

def estimate_transformation(corresponding_points1, corresponding_points2):
    '''
    Estimate the transformation to align the two 3D coordinates systems.
    Calculate the rotation matrix and translation vector that aligns the points from corresponding_points1 to corresponding_points2.
    '''
    # Compute centroids
    centroid1 = np.mean(corresponding_points1, axis=0)
    centroid2 = np.mean(corresponding_points2, axis=0)

    # Center the points
    centered_points1 = corresponding_points1 - centroid1
    centered_points2 = corresponding_points2 - centroid2

    # Compute covariance matrix
    covariance_matrix = centered_points2.T @ centered_points1

    # Perform Singular Value Decomposition
    U, _, Vt = np.linalg.svd(covariance_matrix)

    # Calculate the rotation matrix
    rotation_matrix = Vt @ U.T

    # Calculate the translation vector
    translation_vector = np.mean(corresponding_points2 - corresponding_points1 @ rotation_matrix, axis=0)

    return rotation_matrix, translation_vector

def apply_transformation(rotation_matrix, translation_vector, points):
    '''
    Apply the transformation to the points
    '''
    pass


def compare_poses(path1, path2, angular_error_max_dist=15, translation_error_max_dist=0.25):
    '''
    Compare poses from two different reconstructions using the angular error and the L2 norm of the translation vector
    '''
    if (Path(path1) / "images.bin").is_file() and (Path(path2) / "images.bin").is_file(): 
        # Read COLMAP poses
        rec1 = pycolmap.Reconstruction(path1)
        images1 = rec1.images
        cameras1 = rec1.cameras

        rec2 = pycolmap.Reconstruction(path2)
        images2 = rec2.images
        cameras2 = rec2.cameras

        images1_byname = image_name_asid(images1)
        images2_byname = image_name_asid(images2)
        # Find the common images between the two reconstructions
        images_name_comm = list(set(images1_byname.keys()).intersection(images2_byname.keys()))

        # Principal point of the two cameras sorted by image filename
        camera1_pp = [[float(p) for p in cameras1[images1_byname[i].camera_id].params_to_string().split(",")[-2:]] for i in images_name_comm]
        camera2_pp = [[float(p) for p in cameras2[images2_byname[i].camera_id].params_to_string().split(",")[-2:]] for i in images_name_comm]

        # Align the 3D coordinates space of the two reconstructions, if needed
        if camera1_pp != camera2_pp:
            # Align the estimated poses to the ground truth (or gold standard) poses
            # First estimate transformation, then apply it to the poses
            if len(cameras1) < len(cameras2):
                estimated_rotation, estimated_translation = estimate_transformation(camera1_pp, camera2_pp)
                images1_byname = apply_transformation(estimated_rotation, estimated_translation, images1_byname)
            else:
                estimated_rotation, estimated_translation = estimate_transformation(camera2_pp, camera1_pp)
                images2_byname = apply_transformation(estimated_rotation, estimated_translation, images2_byname)
            breakpoint()

        angular_errors = []
        l2_translations = []
        scene_scale = []
        l2_translations_rescale = []
        for i in images_name_comm:
            angular_errors.append(compute_angular_error(images1_byname[i].rotmat(), images2_byname[i].rotmat()))
            l2_translations_i, scene_scale_i = compute_l2_translation(images1_byname[i].tvec, images2_byname[i].tvec)
            l2_translations.append(l2_translations_i)
            scene_scale.append(scene_scale_i)
            l2_translations_rescale.append(l2_translations_i / scene_scale_i)
        
        aggregate_angular_error = report_angular_error(angular_errors, max_distance=angular_error_max_dist)
        aggregate_l2_translation = report_l2_translation(l2_translations, scene_scale, max_distance=translation_error_max_dist)

        aggregate_angular_error_dict = {
            "percentage_max_dist": aggregate_angular_error, 
            "mean": np.mean(angular_errors),
            "median": np.median(angular_errors),
            "std": np.std(angular_errors),}
        
        aggregate_translation_error_dict = {
            "percentage_max_dist": aggregate_l2_translation, 
            "mean": np.mean(l2_translations),
            "median": np.median(l2_translations),
            "std": np.std(l2_translations),
            "mean_rescale": np.mean(l2_translations_rescale),
            "median_rescale": np.median(l2_translations_rescale),
            "std_rescale": np.std(l2_translations_rescale),}

        print(f"Percentage of Angular Error within {angular_error_max_dist} degree: {aggregate_angular_error}")
        print(f"Percentage of cameras within {translation_error_max_dist*100} percent of the scale of the scene: {aggregate_l2_translation}")

        return aggregate_angular_error_dict, aggregate_translation_error_dict
    else:
        print("One or both reconstructions do not exist")
        return None, None