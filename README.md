
## Requirements

Python 3.8 or later with all [requirements.txt] dependencies installed, including `torch>=1.7`. To install run:

实验用到的模型位于models/cw_yolo路径下；
训练得到的权重位于weights目标下；
beta-CIOU损失位于utils/general.py，第701行bbox_alpha_iou处

```bash
$ pip install -r requirements.txt



## Training

$ python train_alpha.py --data coco.yaml --cfg models/cw_yolo/cw_yolos.yaml --weights '' --batch-size 64
