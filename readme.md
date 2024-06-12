### SAR (Synthetic Aperture Radar) Satellite Ship Detection
git clone the YOLOv5 github into directory and setup


<img src="annotation.png" width=50%>

Dataset Processing
- Using the HRSID dataset
- `python test_train_split.py train`
- `python test_train_split.py test`

View SAR image with annotated ships
- `python describe_image.py train/0`

Training
- `python yolov5/train.py --img 640 --batch 16 --epochs 5 --data dataset.yaml --weights yolov5s.pt --device=0`
- Set device to CUDA GPU if available (requires pytorch/torchvision setup)

Testing
- `python yolov5/detect.py --weights yolov5/runs/train/exp/weights/best.pt --img 640 --conf 0.4 --source yolov5_test/<image name>.jpg`

Todo
- Denoising using deep despeckling network approach (https://arxiv.org/pdf/2110.13148)
- Custom optimizations of yolov5