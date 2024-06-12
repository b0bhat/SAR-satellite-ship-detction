### SAR (Synthetic Aperture Radar) Satellite Ship Detection
git clone the YOLOv5 github into directory

Run `python yolov5/train.py --img 640 --batch 16 --epochs 5 --data dataset.yaml --weights yolov5s.pt --device=0`
(Set device to CUDA GPU if available)