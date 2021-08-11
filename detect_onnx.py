"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def dist(p, q):
    return math.sqrt(sum((px - qx)**2.0 for px, qx in zip(p, q)))


def parse_event(unique_classes, fruit_indices, name_indices):
    # Print results
    event_pred = None
    if (
        fruit_indices.intersection(unique_classes)
        and name_indices["table"] in unique_classes
    ):
        event_pred = "fruit"
    elif (
        name_indices["cart"] in unique_classes
        or name_indices["wok"] in unique_classes
    ) and (
        name_indices["gas_cylinder"] in unique_classes
        or name_indices["burner"] in unique_classes
    ) or (
        name_indices["cart"] in unique_classes
        and name_indices["wok"] in unique_classes
    ):
        event_pred = "chestnut"
    elif (
        name_indices["wheelchair"] in unique_classes
        and name_indices["tissue"] in unique_classes
    ):
        event_pred = "tissue"
    return event_pred


def box_distance(box1, box2):
    """Calculations done w.r.t. box1. Coords are (top left), (bottom right)
    """
    left = box2[2] < box1[0]
    right = box1[2] < box2[0]
    bottom = box2[1] > box1[3]
    top = box1[1] > box2[3]
    if top and left:
        return dist((box1[0], box1[1]), (box2[2], box2[3]))
    elif left and bottom:
        return dist((box1[0], box1[3]), (box2[2], box2[1]))
    elif bottom and right:
        return dist((box1[2], box1[3]), (box2[0], box2[1]))
    elif right and top:
        return dist((box1[2], box1[1]), (box2[0], box2[3]))
    elif left:
        return box1[0] - box2[2]
    elif right:
        return box2[0] - box1[2]
    elif bottom:
        return box2[1] - box1[3]
    elif top:
        return box1[1] - box2[3]
    else:
        # rectangles intersect
        return 0.0


def filter_conf_thres(det, name_indices):
    conf_thres = {
        "apple": 0.22,
        "banana": 0.448,
        "box": 1.0,
        "burner": 0.73,
        "cart": 0.26,
        "gas_cylinder": 0.63,
        "orange": 0.27,
        "table": 0.16,
        "tissue": 0.485,
        "wheelchair": 0.715,
        "wok": 0.45,
    }
    for cls, thres in conf_thres.items():
        det = det[
            (det[..., 5] != name_indices[cls])
            | (
                (det[..., 5] == name_indices[cls])
                & (det[..., 4] > thres)
            )
        ]
    return det


def filter_distance(det, name_indices):
    cond = [True] * len(det)
    distance_thres = {
        "cart": {
            "gas_cylinder": 1.0,
            "burner": 0.0,
        },
        "table": {
            "apple": 1.0,
            "banana": 1.0,
            "orange": 1.0,
        },
        "wheelchair": {
            "tissue": 2.0
        },
        "wok": {
            "burner": 0.1,
        }
    }
    # Loop through reference classes
    for ref_cls in distance_thres:
        ref_dets = det[det[..., 5] == name_indices[ref_cls]]
        # Loop through all detections
        for i, (*det_xyxy, _, det_cls) in enumerate(det):
            det_w = det_xyxy[2] - det_xyxy[0]
            det_h = det_xyxy[3] - det_xyxy[1]
            # Loop through each class related to the reference class
            for cls in distance_thres[ref_cls]:
                if det_cls == name_indices[cls]:
                    coeff = distance_thres[ref_cls][cls]
                    in_proximity = False
                    # Loop through each reference object
                    for *ref_xyxy, _, _ in ref_dets:
                        if (
                            box_distance(ref_xyxy, det_xyxy) <= coeff * det_w
                            and box_distance(ref_xyxy, det_xyxy) <= coeff * det_h
                        ):
                            in_proximity = True
                            break
                    cond[i] = (in_proximity or not len(ref_dets)) and cond[i]
    det = det[cond]
    return det


def filter_location(det, name_indices, img_size):
    lower_y_thres = {
        "cart": 0.5,
        "table": 0.75,
        "wheelchair": 0.5,
        "wok": 0.5,
    }
    for cls, thres in lower_y_thres.items():
        det = det[
            (det[..., 5] != name_indices[cls])
            | (
                (det[..., 5] == name_indices[cls])
                & (det[..., 3] > thres * img_size[0])
            )
        ]
    return det


def filter_rel_size(det, name_indices):
    cond = [True] * len(det)
    size_thres = {
        "cart": {
            # "gas_cylinder": 1.0,
            "burner": 0.513,
        },
        "table": {
            "apple": 0.475,
            "banana": 0.613,
            "orange": 0.265,
        },
        "wheelchair": {
            "tissue": 0.385
        },
    }
    # Loop through reference classes
    for ref_cls in size_thres:
        ref_dets = det[det[..., 5] == name_indices[ref_cls]]
        # Loop through all detections
        for i, (*det_xyxy, _, det_cls) in enumerate(det):
            det_size = (det_xyxy[2] - det_xyxy[0]) * (det_xyxy[3] - det_xyxy[1])
            # Loop through each class related to the reference class
            for cls in size_thres[ref_cls]:
                if det_cls == name_indices[cls]:
                    coeff = size_thres[ref_cls][cls]
                    apt_size = False
                    # Loop through each reference object
                    for *ref_xyxy, _, _ in ref_dets:
                        ref_size = (ref_xyxy[2] - ref_xyxy[0]) * (ref_xyxy[3] - ref_xyxy[1])
                        if ref_size * coeff > det_size:
                            apt_size = True
                            break
                    cond[i] = apt_size or not len(ref_dets)
    det = det[cond]
    return det


def filter_size(det, name_indices, img_size):
    height_thres = {
        "apple": 0.2,
        "banana": 0.2,
        "table": 0.5,
        "orange": 0.2,
    }
    for cls, thres in height_thres.items():
        det = det[
            (det[..., 5] != name_indices[cls])
            | (
                (det[..., 5] == name_indices[cls])
                & (det[..., 3] - det[..., 1] <= thres * img_size[0])
            )
        ]
    return det


@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        # visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        ):
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, pt, onnx = False, w.endswith('.pt'), w.endswith('.onnx')  # inference type
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        # check_requirements(('onnx', 'onnxruntime'))
        import onnxruntime
        session = onnxruntime.InferenceSession(w, None)
        names = ["apple", "banana", "orange", "wheelchair", "wok", "box", "table", "tissue", "gas_cylinder", "burner", "cart"]
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # initialize_class_indices
    name_indices = {name: i for i, name in enumerate(names)}
    fruit_color = len(name_indices)
    fruit_names = ["apple", "banana", "orange"]
    fruit_indices = set([name_indices[name] for name in fruit_names])
    event_indices = {
        "fruit": [
            name_indices["apple"],
            name_indices["banana"],
            name_indices["orange"],
            name_indices["box"],
            name_indices["table"],
        ],
        "chestnut": [
            name_indices["wok"],
            name_indices["gas_cylinder"],
            name_indices["burner"],
            name_indices["cart"],
        ],
        "tissue": [name_indices["wheelchair"], name_indices["tissue"]],
    }
    event_color_indices = {
        event: i for i, event in enumerate(["fruit", "chestnut", "tissue"])
    }

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        if pt:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        elif onnx:
            img = img.astype('float32')
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        t1 = time_synchronized()
        if pt:
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            # pred = model(img, augment=augment, visualize=visualize)[0]
            pred = model(img, augment=augment)[0]
        elif onnx:
            pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_synchronized()

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        event_path = save_dir / "event.txt"
        # Process predictions
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            if len(det):
                img_size = img.shape[2:]
                det = filter_conf_thres(det, name_indices)
                det = filter_location(det, name_indices, img_size)
                det = filter_size(det, name_indices, img_size)
                det = filter_distance(det, name_indices)
                det = filter_rel_size(det, name_indices)
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img_size, det[:, :4], im0.shape).round()

                # Parse event
                unique_classes = set([int(cls.item()) for cls in det[:, -1].unique()])
                event_pred = parse_event(unique_classes, fruit_indices, name_indices)
                
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                coords = []
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if event_pred and cls in event_indices[event_pred]:
                        coords.append(torch.tensor(xyxy).view(1, 4).view(-1).tolist())
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        color = colors(c, True)
                        if any(name in label for name in fruit_names):
                            label = f"fruit {label.split(' ')[-1]}"
                            color = colors(fruit_color, True)
                        plot_one_box(xyxy, im0, label=label, color=color, line_thickness=line_thickness)
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                if event_pred:
                    with open(event_path, "a") as f:
                        f.write(f"{p.name} {event_pred}\n")
                    coords = np.asarray(coords)
                    min_xy = np.amin(coords[:, :2], axis=0)
                    max_xy = np.amax(coords[:, 2:], axis=0)
                    event_xyxy = (min_xy[0], min_xy[1], max_xy[0], max_xy[1])
                    event_label = f"{event_pred} hawker"
                    plot_one_box(
                        event_xyxy,
                        im0,
                        label=event_label,
                        color=colors(event_color_indices[event_pred], True),
                        line_thickness=opt.line_thickness,
                    )
            
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(f'Done. ({time.time() - t0:.3f}s)')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    # parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    return opt


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
