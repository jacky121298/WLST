#!/bin/bash
for i in $(seq 0 10 50)
do
    # python ./utils/ps_eval.py --ps ../output/da-waymo-kitti_models/pvrcnn_st3d/pvrcnn_st3d/default/ps_label/ps_label_e"$i".pkl --ps_from st3d
    python ./utils/ps_eval.py --ps ../output/da-waymo-kitti_models/pvrcnn_st3d/wl_labeler_ros_st/consistency/ps_label/ps_label_e"$i"_M3D.pkl --ps_from wlst
done