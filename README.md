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