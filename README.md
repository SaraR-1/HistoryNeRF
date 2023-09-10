## 0. Installation: Setup the environment
Use the provided docker image to install the dependencies for nerfstudio. Docker (get docker) and nvidia GPU drivers (get nvidia drivers), capable of working with CUDA 11.8, must be installed.

Then update the required dependencies:
```
python3.10 -m pip install --upgrade pip
pip install -e ./

```

## 1. Run COLMAP only on the entire dataset
The input path can either be a folder to the images of a video that will be divided into frames before running the pose estimation
```
python3 historynerf/run_goldstandard.py input_dir=/workspace/data/old_petrol_pumps/data/video_trim1.mp4 output_dir=/workspace/data/old_petrol_pumps/data pose_estimation.matching_method=sequential
```

```
python3 historynerf/run_goldstandard.py input_dir=/workspace/data/old_petrol_pumps/data/video_trim1.mp4 output_dir=/workspace/data/old_petrol_pumps/data pose_estimation.matching_method=exhaustive
```

## 2. Split the data into train and test, this will also split the COLMAP previously estimated accordingly 
```
python3 historynerf/run_splitdata.py camera_path=/workspace/data/old_petrol_pumps/processed_data/transforms.json n=80 images_dir=/workspace/data/old_petrol_pumps/processed_data/images output_dir=/workspace/data/old_petrol_pumps/data
```

## 3. Run NeRFs experiments
Run the help function to visualise all the arguments:
```
python3 historynerf/run.py --help
```

Let's now see different example of experiments we can run
### Undersample, Run COLMAP, Train a NeRF model

### Use previously undersampled, use previously run COLMAP, Train a NeRF model
Note that two types of COLMAP results are accepted here:
a. COLMAP run on the same undersampled set (this is useful if I want to only use a smaller subset, but only change something in the NeRF training).
b. COLMAP run on the gold standard dataset (in step 1., before splitting into train and test and undersampling the training set). This is useful to compare how NeRF's results depend on COLMAP. This is based on the assumption the COLMAP estimated on the entire set are more accurate than the camera pose estimated only using a smaller set of the training data.