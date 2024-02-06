# Getting Started

The dataset configs are located within [tools/cfgs/dataset_configs](../tools/cfgs/dataset_configs), and the model configs are located within [tools/cfgs](../tools/cfgs) for different datasets. 

### KITTI Dataset

* Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows (the road planes could be downloaded from [[road plane]](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing), which are optional for data augmentation in the training):

* NOTE: if you already have the data infos from `pcdet v0.1`, you can choose to use the old infos and set the DATABASE_WITH_FAKELIDAR option in tools/cfgs/dataset_configs/kitti_dataset.yaml as True. The second choice is that you can create the infos and gt database again and leave the config unchanged.

```
OpenPCDet
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
├── pcdet
├── tools
```

* Generate the data infos by running the following command:

```python
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

### NuScenes Dataset

* Please download the official [NuScenes 3D object detection dataset](https://www.nuscenes.org/download) and organize the downloaded files as follows:

```
OpenPCDet
├── data
│   ├── nuscenes
│   │   │── v1.0-trainval (or v1.0-mini if you use mini)
│   │   │   │── samples
│   │   │   │── sweeps
│   │   │   │── maps
│   │   │   │── v1.0-trainval  
├── pcdet
├── tools
```

* Install the `nuscenes-devkit` with version `1.1.9` by running the following command:

```shell script
pip install nuscenes-devkit==1.1.9
```

* Generate the data infos by running the following command (it may take several hours):

```python
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml --version v1.0-trainval
```

### Waymo Open Dataset

* Please download the official [Waymo Open Dataset](https://waymo.com/open/download/), including the training data `training_0000.tar~training_0031.tar` and the validation data `validation_0000.tar~validation_0007.tar`.

* Unzip all the above `xxxx.tar` files to the directory of `data/waymo/raw_data` as follows (You could get 798 *train* tfrecord and 202 *val* tfrecord ):  

```
OpenPCDet
├── data
│   ├── waymo
│   │   │── ImageSets
│   │   │── raw_data
│   │   │   │── segment-xxxxxxxx.tfrecord
|   |   |   |── ...
|   |   |── waymo_processed_data
│   │   │   │── segment-xxxxxxxx/
|   |   |   |── ...
│   │   │── pcdet_gt_database_train_sampled_xx/
│   │   │── pcdet_waymo_dbinfos_train_sampled_xx.pkl   
├── pcdet
├── tools
```

* Install the official `waymo-open-dataset` by running the following command:

```shell script
pip3 install --upgrade pip
# tf 2.1.0
pip3 install waymo-open-dataset-tf-2-1-0==1.2.0
```

* Extract point cloud data from tfrecord and generate data infos by running the following command (it takes several hours, and you could refer to `data/waymo/waymo_processed_data` to see how many records that have been processed):

```python
python -m pcdet.datasets.waymo.waymo_dataset_wl --func create_waymo_infos_wl --cfg_file tools/cfgs/dataset_configs/da_waymo_dataset_wl.yaml
```

Note that you do not need to install `waymo-open-dataset` if you have already processed the data before and do not need to evaluate with official Waymo Metrics. 

## Training & Testing

### Pre-train 3D Detector

You could optionally add extra command line parameters `--batch_size ${BATCH_SIZE}` and `--epochs ${EPOCHS}` to specify your preferred parameters.

* Train with multiple GPUs or multiple machines

```shell script
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE}

# or

sh scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} ${NUM_GPUS} --cfg_file ${CONFIG_FILE}
```

* Train with a single GPU:

```shell script
python train.py --cfg_file ${CONFIG_FILE}
```

### Pre-train Autolabeler

1. Train segmentation model for autolabeler

```shell script
python train_wlst_seg.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}
```

2. Train regression model for autolabeler

```shell script
python train_wlst.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --seg_model ${SEG_MODEL_PATH}
```

### Train WLST

* Self-training process:

```shell script
python train_wlst.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --pretrained_model ${AUTOLABELER_PATH} --pretrained_model_M3D ${DETECTOR_PATH}
```

### Testing

* Test with a pretrained model:

```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
```

* To test all the saved checkpoints of a specific training setting and draw the performance curve on the Tensorboard, add the `--eval_all` argument:

```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --eval_all
```

* To test with multiple GPUs:

```shell script
sh scripts/dist_test.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}

# or

sh scripts/slurm_test_mgpu.sh ${PARTITION} ${NUM_GPUS} --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}
```