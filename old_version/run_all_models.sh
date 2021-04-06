#CUDA_VISIBLE_DEVICES=1 python train_multi.py cfg/occlusion.data cfg/yolo-pose-multi.cfg ../dataset/backup/init.weights
#CUDA_VISIBLE_DEVICES=1 python train_multi.py cfg/occlusion.data cfg/yolo-pose-multi.cfg ../checkpoint_multi/yolo-pose-multi/model.weights
python train_multi.py cfg/occlusion.data cfg/yolom-pose-pre.cfg ../checkpoint_multi/yolom-pose-pre/init.weights