## NeRFStudio Preprocessing and Structure from motion
Usage: `ns-process-data images` `--data PATH --output-dir PATH [--verbose | --no-verbose] [--camera-type {perspective,fisheye,equirectangular}]
                              [--matching-method {exhaustive,sequential,vocab_tree}] [--sfm-tool {any,colmap,hloc}] [--refine-pixsfm | --no-refine-pixsfm]
                              [--feature-type {any,sift,superpoint,superpoint_aachen,superpoint_max,superpoint_inloc,r2d2,d2net-ss,sosnet,disk}]
                              [--matcher-type {any,NN,superglue,superglue-fast,NN-superpoint,NN-ratio,NN-mutual,adalam}] [--num-downscales INT]
                              [--skip-colmap | --no-skip-colmap] [--skip-image-processing | --no-skip-image-processing] [--colmap-model-path PATH] [--colmap-cmd STR]
                              [--images-per-equirect {8,14}] [--crop-factor FLOAT FLOAT FLOAT FLOAT] [--crop-bottom FLOAT] [--gpu | --no-gpu]
                              [--use-sfm-depth | --no-use-sfm-depth] [--include-depth-debug | --no-include-depth-debug]`

Explaining `any` defaults, specifically how sfm replaces the default parameters `any` by usable value:

```
if sfm_tool == "any":
    if (feature_type in ("any", "sift")) and (matcher_type in ("any", "NN")):
        sfm_tool = "colmap"
    else:
        sfm_tool = "hloc"
if sfm_tool == "colmap":
    if (feature_type not in ("any", "sift")) or (matcher_type not in ("any", "NN")):
        return (None, None, None)
    return ("colmap", "sift", "NN")
if sfm_tool == "hloc":
    if feature_type in ("any", "superpoint"):
        feature_type = "superpoint_aachen"

    if matcher_type == "any":
        matcher_type = "superglue"
    elif matcher_type == "NN":
        matcher_type = "NN-mutual"
```
