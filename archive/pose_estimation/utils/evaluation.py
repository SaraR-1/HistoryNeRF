import numpy as np
import pycolmap
from pathlib import Path
import sys
from historynerf.pose_estimation.evaluation import image_name_asid

def check_same_centers(path1, path2):
    '''
    Check if the two reconstructions have the same principal point for all the images
    E.g. 
    path1 = Path("/srv/galene0/sr572/palace_of_westminster/dense/sparse")
    path2 = Path("/srv/galene0/sr572/palace_of_westminster/dense/output")
    path2 = path2 / seed0_nsamples100
    print(check_same_centers(gt_path, path_dir /  sys.argv[1]))
    '''
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

    return camera1_pp == camera2_pp


def unsersample_close_cameras(path, nsamples, seed=None):
    '''
    Select a subset of images based on their cameras distance.
    Args:
        path: path to the reconstruction
        seed: seed to select the first image, if None the first two images are the one with the smallest camera distance across all cameras
        nsamples: number of samples to select
    '''
    rec = pycolmap.Reconstruction(path)
    images = rec.images
    # cameras = rec.cameras

    # Get the translation vector and the corresponding image name for each camera (image)
    transl_vect, names = zip(*[(i.tvec, i.name) for i in images.values()])
    transl_vect, names = np.array(transl_vect), np.array(names)
    # Compute the pairwise distance between all the cameras
    dist = np.linalg.norm(transl_vect[:, None] - transl_vect, axis=-1)
    # Set the diagonal to inf
    np.fill_diagonal(dist, np.inf)
    # Find the two images with the smallest distance
    if seed is None:
        # Select the images with the smallest distance
        idx_first = np.unravel_index(dist.argmin(), dist.shape)[0]
    else:
        # Randomly select the first image
        np.random.seed(seed)
        idx_first = np.random.randint(0, len(images), size=1)[0]
        
    # Select nsamples images (indices) that are the closest from the first image
    idx_closest = np.append(np.argsort(dist[idx_first])[:nsamples-1], idx_first)
    names_closest = [names[i] for i in idx_closest]

    # check = [i for i in images.keys() if images[i].name == names[idx_first]]
    return names_closest


if __name__ == "__main__":
    # path_dir = Path("/srv/galene0/sr572/palace_of_westminster/dense/output")
    gt_path = Path("/srv/galene0/sr572/palace_of_westminster/dense/sparse")
    # print(check_same_centers(gt_path, path_dir /  sys.argv[1]))

    # If only an argument is passed, it is the number of samples
    # If two arguments are passed, the first is the number of samples and the second is the seed
    if len(sys.argv) == 2:
        print(unsersample_close_cameras(gt_path, int(sys.argv[1])))
    else:
     print(unsersample_close_cameras(gt_path, int(sys.argv[1]), int(sys.argv[2])))