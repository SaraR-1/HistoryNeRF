import os

class NSWrapper:
    def __init__(self):
        self.initialize()
    
    def initialize(self):
        '''
        Depending of the NeRFStudio output, it could be good to create output directories within the specified output directory.
        '''
        pass

    def process_data(self):
        '''
        Data Preprocessing (includes pose estimation with COLMAP)
        '''
        # TODO: change as needed
        command = f"ns-process-data {images, video} --data {DATA_PATH} --output-dir {PROCESSED_DATA_DIR}"
        os.system(command)

    def train(self):
        command = f"ns-train nerfacto --data {PROCESSED_DATA_DIR}"
        os.system(command)

    def run(self):
        self._process_data()
        self._train()
