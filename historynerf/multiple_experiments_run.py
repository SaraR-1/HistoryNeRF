from pathlib import Path
import os


def main():
    experiments_path = Path('/workspace/data/bridge_of_sighs/output_colmap_nerf')
    nframes_list = os.listdir(experiments_path)
    nframes_list = [i for i in nframes_list if i not in ['every1frames', 'alldataset']]
    # nframes_list = ['every1frames']

    for nframes in nframes_list:
        print(nframes)
        use_gradient_scaling = True
        print(use_gradient_scaling)
        command = f'python3 historynerf/run.py wandb_project=bridge_of_sighs_colmap_nerf data_preparation.input_dir={experiments_path}/{nframes}/images data_preparation.output_dir={experiments_path}/{nframes} data_preparation.overwrite_output=True pose_estimation.skip_colmap_flag=True pose_estimation.skip_image_processing_flag=True pose_estimation.colmap_model_path={experiments_path}/{nframes}/processed_data_fixedcolmap/colmap/sparse/0 nerf.vis=wandb nerf.max_num_iterations=60000 nerf.train_split_fraction=1. nerf.pipeline.model.use_gradient_scaling={use_gradient_scaling} nerf.method_name=nerfacto nerf.dataparser_name=nerfstudio-data'
        os.system(command)
        # experiment_list = os.listdir(experiments_path / nframes / "nerf")
        # for experiment in experiment_list:
        #     print(experiment)
        #     command = f"python3 historynerf/run_evaluation.py config_path=/workspace/data/bridge_of_sighs/output_colmap_nerf/{nframes}/nerf/{experiment}/nerfacto/default/config.yml camera_pose_path_test=/workspace/data/bridge_of_sighs/data/test/transforms.json gt_images_dir=/workspace/data/bridge_of_sighs/data/test output_dir=/workspace/data/bridge_of_sighs/output_colmap_nerf/{nframes}/nerf/{experiment}/nerfacto/default/evaluation"
        #     os.system(command)



if __name__ == "__main__":
    main()



