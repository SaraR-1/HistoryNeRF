## Run COLMAP
Check hydra config:
```
python historynerf/pose_estimation/run.py -h
```
How to run COLMAP pose estimation:
```
python historynerf/pose_estimation/run.py wandb_project=colmap_realvideos wandb_log=True pose_config.use_gpu=5 pose_config.image_dir=/sheldonian/frames pose_config.output_dir=/sheldonian/output pose_config.matching_method=sequential pose_config.video_sample_step=5
```
**Practical info:** if the images are extracted from a video `sequential` matching should be used, `exhaustive` otherwise.

## COLMAP Create Screencast
First, start the COLMAP GUI by executing:
```
colmap gui
```

Then, import the model using `File > Import Model`. The folder should contain 3 files: cameras.bin, images.bin and points3D.bin.
To create a video screen capture of the reconstructed model, choose `Extras > Grab movie`. This dialog allows to set individual control viewpoints by choosing `Add`.
Save the individual frames of the video capture selecting `Assemble movie`.

The frames can then be assembled to a movie using FFMPEG with the following command:
```
ffmpeg -i frame%06d.png -r 30 -vf scale=1680:1050 movie.mp4
```

For additional information check the [COLMAP documentation](https://colmap.github.io/gui.html).