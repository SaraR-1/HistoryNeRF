import numpy as np
import pycolmap
from pathlib import Path

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


def compute_l2_translation(translation1, translation2):
    '''
    L2 norm between two translation vectors
    Args:
        translation1: 3x1 translation vector
        translation2: 3x1 translation vector
    Returns:
        l2_norm: L2 norm between the two translation vectors'''
    return np.linalg.norm(translation1 - translation2)

def read_colmap_poses(path):
    '''
    Read COLMAP poses
    Args:
        path: path to the directory containing the cameras, images and points3D bin files
    Returns:
        images: pycolmap.Image object
    '''
    reconstruction = pycolmap.Reconstruction(path)
    return reconstruction.images

def match_colmap_poses(poses1, poses2):
    '''
    Match poses from two different reconstructions
    Args:
    Returns:
    '''
    pass

def image_name_asid(images):
    '''
    Use the image name as key for the images dictionary
    '''
    return {i.name:i for i in images.values()}

def compare_poses(path1, path2):
    '''
    Compare poses from two different reconstructions using the angular error and the L2 norm of the translation vector
    '''
    # Read COLMAP poses
    #images1 = read_colmap_poses(path1)
    images2 = read_colmap_poses(path2)   
    breakpoint() 
    # prova = pycolmap.Reconstruction(path2).images
    # rec = pycolmap.Reconstruction(path2)
    # pr = rec.images

    images1_byname = image_name_asid(images1)
    images2_byname = image_name_asid(images2)
    # Find the common images between the two reconstructions
    images_name_comm = list(set(images1_byname.keys()).intersection(images2_byname.keys()))

    angular_errors = []
    l2_translations = []
    for i in images_name_comm:
        angular_errors.append(compute_angular_error(images1_byname[i].rotmat(), images2_byname[i].rotmat()))
        l2_translations.append(compute_l2_translation(images1_byname[i].tvec, images2_byname[i].tvec))


if __name__ == "__main__":
    compare_poses(path1="/Users/sr572/Documents/ScienceMuseum/Datasets/trevi_fountain/dense/sparse", 
                  path2="/Users/sr572/Documents/ScienceMuseum/Datasets/toy_trevi_fountain/output")