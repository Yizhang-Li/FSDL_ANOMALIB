from anomalib.data import MVTec
datamodule = MVTec(root="./datasets/MVTec",category="leather",image_size=256)
datamodule.prepare_data()