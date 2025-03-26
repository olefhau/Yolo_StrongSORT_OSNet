import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from PIL import Image
from ultralytics import YOLO  # Use ultralytics for YOLO model inference
import cv2

# Limit the number of CPUs used by high-performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # Add ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # Add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # Relative path

from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT


# Method for saving cropped images
def save_one_box(xyxy, im, file=Path('im.jpg'), gain=1.02, pad=10, square=False, BGR=False, save=True):
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # Convert to xywh format
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # Make square
    b[:, 2:] = b[:, 2:] * gain + pad  # Apply gain and padding
    xyxy = xywh2xyxy(b).long()
    clip_coords(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)  # Create directory
        f = str(Path(file).with_suffix('.jpg'))
        Image.fromarray(crop).save(f, quality=95, subsampling=0)
    return crop


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    Rescale coordinates (x1, y1, x2, y2) from the model's input image size to the original image size.
    Args:
        img1_shape (tuple): Shape of the model's input image (height, width).
        coords (torch.Tensor): Bounding box coordinates in the format (x1, y1, x2, y2).
        img0_shape (tuple): Shape of the original image (height, width).
        ratio_pad (tuple): Optional ratio and padding values (ratio, (pad_x, pad_y)).
    Returns:
        torch.Tensor: Rescaled bounding box coordinates.
    """
    if ratio_pad is None:  # Calculate from shapes
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # Gain (wh ratio)
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # Wh padding
    else:
        gain = ratio_pad[0]
        pad = ratio_pad[1]

    # Perform out-of-place operations
    coords = coords.clone()  # Clone the tensor to avoid in-place operations
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].clamp(0, max(img0_shape))  # Clip coordinates to image size
    return coords

def xyxy2xywh(x):
    """
    Convert bounding box format from [x1, y1, x2, y2] to [x_center, y_center, width, height].
    Args:
        x (torch.Tensor): Bounding box coordinates in [x1, y1, x2, y2] format.
    Returns:
        torch.Tensor: Bounding box coordinates in [x_center, y_center, width, height] format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x_center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y_center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def plot_one_box(x, img, color=(255, 0, 0), label=None, line_thickness=3):
    """
    Draws a single bounding box on an image.
    Args:
        x (list or np.ndarray): Bounding box coordinates in [x1, y1, x2, y2] format.
        img (np.ndarray): Image on which to draw the bounding box.
        color (tuple): Color of the bounding box in (B, G, R) format.
        label (str): Optional label to display on the bounding box.
        line_thickness (int): Thickness of the bounding box lines.
    """
    # Convert coordinates to integers
    x1, y1, x2, y2 = map(int, x)
    # Draw the rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=line_thickness)
    # Add label if provided
    if label:
        font_scale = max(0.5, line_thickness / 3)
        font_thickness = max(1, line_thickness // 3)
        t_size = cv2.getTextSize(label, 0, font_scale, font_thickness)[0]
        label_y1 = max(y1 - t_size[1] - 3, 0)
        label_y2 = y1
        label_x1 = x1
        label_x2 = x1 + t_size[0] + 3
        cv2.rectangle(img, (label_x1, label_y1), (label_x2, label_y2), color, -1)  # Filled rectangle for label
        cv2.putText(img, label, (x1, y1 - 2), 0, font_scale, (255, 255, 255), thickness=font_thickness, lineType=cv2.LINE_AA)

VID_FORMATS = ('asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv')  # Video formats


@torch.no_grad()
def run(
        source='0',
        yolo_model='yolov8s.pt',  # Specify the YOLO model name (e.g., yolov8n.pt, yolov8s.pt)
        strong_sort_weights=WEIGHTS / 'osnet_x1_0_msmt17.pt',  # StrongSORT weights
        config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
        imgsz=(640, 640),  # Inference size (height, width)
        conf_thres=0.25,  # Confidence threshold
        iou_thres=0.45,  # NMS IoU threshold
        max_det=1000,  # Maximum detections per image
        device='',  # CUDA device, i.e., 0 or 0,1,2,3 or CPU
        show_vid=True,  # Show results
        save_txt=False,  # Save results to *.txt
        save_conf=False,  # Save confidences in --save-txt labels
        save_crop=False,  # Save cropped prediction boxes
        save_vid=False,  # Save video results
        nosave=False,  # Do not save images/videos
        classes=None,  # Filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # Class-agnostic NMS
        augment=False,  # Augmented inference
        visualize=False,  # Visualize features
        update=False,  # Update all models
        project=ROOT / 'runs/track',  # Save results to project/name
        name='exp',  # Save results to project/name
        exist_ok=False,  # Existing project/name ok, do not increment
        line_thickness=3,  # Bounding box thickness (pixels)
        hide_labels=False,  # Hide labels
        hide_conf=False,  # Hide confidences
        hide_class=False,  # Hide IDs
        half=False,  # Use FP16 half-precision inference
        dnn=False,  # Use OpenCV DNN for ONNX inference
):

    # Set the device if not specified
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)  # Convert to torch.device

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # Save inference images
    is_file = Path(source).suffix[1:] in VID_FORMATS
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # Download

    # Directories
    exp_name = name if name else Path(yolo_model).stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # Increment run
    save_dir = Path(save_dir)
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # Make directory

    # Load YOLO model using ultralytics
    model = YOLO(yolo_model)  # Automatically downloads the model if not available locally

    # Initialize StrongSORT
    cfg = get_config()
    cfg.merge_from_file(config_strongsort)

    strongsort_list = []
    for i in range(1):  # Assuming single source for simplicity
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                half,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,
            )
        )
        strongsort_list[i].model.warmup()

    # Define dataset
    if webcam:
        # Webcam or video stream
        dataset = cv2.VideoCapture(int(source) if source.isnumeric() else source)
    elif is_file:
        # Single video file
        dataset = cv2.VideoCapture(source)
    else:
        raise ValueError(f"Unsupported source type: {source}")
    
    # Run tracking
    while dataset.isOpened():
        ret, im0s = dataset.read()
        if not ret:
            break

        # Preprocess image
        h0, w0 = im0s.shape[:2]  # Original height and width
        r = min(imgsz[0] / h0, imgsz[1] / w0)  # Resize ratio (new / old)

        # Compute padding
        new_unpad = int(round(w0 * r)), int(round(h0 * r))
        dw, dh = imgsz[1] - new_unpad[0], imgsz[0] - new_unpad[1]  # Wh padding
        dw, dh = dw // 2, dh // 2  # Divide padding into 2 sides

        # Resize and pad image
        im = cv2.resize(im0s, new_unpad, interpolation=cv2.INTER_LINEAR)    
        im = cv2.copyMakeBorder(im, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # Add padding
        im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        im = np.ascontiguousarray(im)

        im = torch.from_numpy(im).to(device)
        im = im.float() / 255.0  # Normalize to 0.0 - 1.0

        # Inference
        results = model(im, conf=conf_thres, iou=iou_thres, classes=classes, agnostic_nms=agnostic_nms)

        # Process detections
        for i, det in enumerate(results):  # Iterate over predictions
            det = det.boxes.data  # Access the bounding boxes, confidence, and class labels
            if det is not None and len(det):

                # Clone the tensor to avoid in-place updates
                det_clone = det.clone()

                # Rescale boxes from YOLO input size to original image size
                det_clone[:, :4] = scale_coords((imgsz[0], imgsz[1]), det_clone[:, :4], im0s.shape[:2], (r, (dw, dh))).round()

                # Extract bounding boxes, confidences, and class labels
                xywhs = xyxy2xywh(det_clone[:, 0:4])
                confs = det_clone[:, 4]
                clss = det_clone[:, 5]

                # Pass detections to StrongSORT
                outputs = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0s)

                # Draw boxes and labels
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]

                        label = f"{id} {model.names[int(cls)]} {conf:.2f}"
                        plot_one_box(bboxes, im0s, label=label, color=(255, 0, 0), line_thickness=2)

        # Show video if enabled
        if show_vid:
            cv2.imshow('Tracking', im0s)
            if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
                break

    dataset.release()
    cv2.destroyAllWindows()

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    """
    Increment a file or directory path, i.e. runs/exp --> runs/exp1, runs/exp2, etc.
    Args:
        path (str or Path): Path to increment.
        exist_ok (bool): If True, existing paths are allowed and no incrementing is done.
        sep (str): Separator to use between the base name and the incrementing number.
        mkdir (bool): If True, create the directory after incrementing.
    Returns:
        Path: Incremented path.
    """
    path = Path(path)
    if path.exists() and not exist_ok:
        base, suffix = path.stem, path.suffix
        for n in range(1, 1000):  # Limit to 1000 increments
            new_path = path.parent / f"{base}{sep}{n}{suffix}"
            if not new_path.exists():
                if mkdir:
                    new_path.mkdir(parents=True, exist_ok=True)
                return new_path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', nargs='+', type=str, default='yolov8n.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # Expand
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
