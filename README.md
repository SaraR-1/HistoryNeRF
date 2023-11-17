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
python3 historynerf/run_splitdata.py camera_path=/workspace/data/old_petrol_pumps/data/processed_data/transforms.json n=80 images_dir=/workspace/data/old_petrol_pumps/data/processed_data/images output_dir=/workspace/data/old_petrol_pumps/data
```

## 3. Run NeRFs experiments
Run the help function to visualise all the arguments:
```
python3 historynerf/run.py --help
```

Let's now see different example of experiments we can run
### Undersample, Run COLMAP, Train a NeRF model
```python
python3 historynerf/run.py wandb_log=True wandb_project=old_petrol_pumps data_preparation.input_dir=/workspace/data/old_petrol_pumps/data/train/images data_preparation.output_dir=/workspace/data/old_petrol_pumps/every50frames data_preparation.overwrite_output=False data_preparation.sampling.sequential_sample_step=50 pose_estimation.matching_method=exhaustive nerf.train_split_fraction=1. nerf.pipeline.model.use_gradient_scaling=True nerf.vis=wandb nerf.max_num_iterations=60000 evaluation.alignment.flag=True
```

### Use previously undersampled, use previously run COLMAP, Train a NeRF model
Note that two types of COLMAP results are accepted here:
a. COLMAP run on the same undersampled set (this is useful if I want to only use a smaller subset, but only change something in the NeRF training).
b. COLMAP run on the gold standard dataset (in step 1., before splitting into train and test and undersampling the training set). This is useful to compare how NeRF's results depend on COLMAP. This is based on the assumption the COLMAP estimated on the entire set are more accurate than the camera pose estimated only using a smaller set of the training data.


## Using Gaussian-splatting
1. Clone the repo
```
cd HistoryNeRF
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
```
2. Anaconda is required
```
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
bash Anaconda3-2023.03-1-Linux-x86_64.sh
```
3. Install and Activate Environment
```
cd gaussian-splatting
conda env create --file environment.yml
conda activate gaussian_splatting
```
4. Help!
```
python train.py --help
```
5. Basic Run
We can take advantage of the COLMAP estimated using our NeRFStudio wrapper. But this will require some adjustement. 
A. 
The following dataset structure is expected in the source path location:
```
<location>
|---input
|   |---<image 0>
|   |---<image 1>
|   |---...
|---distorted
    |---database.db
    |---sparse
        |---0
            |---...
```
We can do it using:
```
python ../gaussian_splatting_restructure_folders.py /workspace/data/old_petrol_pumps/every5frames/processed_data /workspace/data/old_petrol_pumps/every5frames/processed_data_for_gaussian
```
B. We need to undistor the images, we can run:
```
python convert.py --source_path /workspace/data/old_petrol_pumps/every5frames/processed_data_for_gaussian --skip_matching
```
C. We can finally train our model
```
python train.py --source_path /workspace/data/old_petrol_pumps/every5frames/processed_data_for_gaussian --model_path /workspace/data/old_petrol_pumps/every5frames/gaussian-splatting --iterations 60000
```
N.B. you can render it, but it will just produce the rendered images, no video.
```
python render.py -m /workspace/data/old_petrol_pumps/every5frames/gaussian-splatting --source_path /workspace/data/old_petrol_pumps/every5frames/processed_data_for_gaussian --skip_test True --skip_train False
```
6. Web Render
Use this online web renderer, way easier!! 
```
https://antimatter15.com/splat/....
```