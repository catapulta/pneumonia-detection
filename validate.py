from __future__ import print_function
import argparse
import os
import numpy as np
import torch
from torchvision import transforms
import dataset
from darknet import Darknet
from utils import get_all_boxes, nms, read_data_cfg, logging, map_iou

# etc parameters
use_cuda = True
seed = 22222
eps = 1e-5

FLAGS = None


def main():
    # Validation parameters
    conf_thresh = FLAGS.conf_threshold
    nms_thresh = FLAGS.nms_threshold
    iou_thresh = FLAGS.iou_threshold

    # output file
    out_path = FLAGS.out_path

    # Training settings
    datacfg = FLAGS.data
    cfgfile = FLAGS.config

    data_options = read_data_cfg(datacfg)
    file_list = data_options['valid']
    gpus = data_options['gpus']  # e.g. 0,1,2,3
    ngpus = len(gpus.split(','))

    num_workers = int(data_options['num_workers'])
    # for testing, batch_size is set to 1 (one)
    batch_size = FLAGS.batch_size

    global use_cuda
    use_cuda = torch.cuda.is_available() and use_cuda

    ###############
    torch.manual_seed(seed)
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)

    global model
    model = Darknet(cfgfile)
    # model.print_network()

    init_width = model.width
    init_height = model.height

    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}

    val_loader = torch.utils.data.DataLoader(
        dataset.listDataset(file_list, shape=(init_width, init_height),
                            shuffle=False, jitter=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                            ]), validate=True),
        batch_size=batch_size, shuffle=False, **kwargs)

    if use_cuda:
        if ngpus > 1:
            model = torch.nn.DataParallel(model)
            model = model.module
    model = model.to(torch.device("cuda" if use_cuda else "cpu"))
    for w in FLAGS.weights:
        # model.load_weights(w)
        checkpoint = torch.load(w)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging('evaluating ... %s' % (w))
        test(val_loader, conf_thresh, nms_thresh, iou_thresh, out_path, batch_size)


def test(val_loader, conf_thresh, nms_thresh, iou_thresh, out_path, batch_size):
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i
        return 50

    model.eval()
    num_classes = model.num_classes
    device = torch.device("cuda" if use_cuda else "cpu")

    if model.net_name() == 'region':  # region_layer
        shape = (0, 0)
    else:
        shape = (model.width, model.height)
    map = []
    for i, (imgpath, data, target, org_w, org_h) in enumerate(val_loader):
        print('Computing boxes for batch', i, 'of size', batch_size, '. Number computed is:', i * batch_size)
        data = data.to(device)
        output = model(data)
        all_boxes, det_confs = get_all_boxes(output, shape, conf_thresh, num_classes, use_cuda=use_cuda,
                                             output_confidence=True)

        for k in range(len(all_boxes)):
            boxes = np.array(all_boxes[k])
            boxes = boxes[boxes[:, 4] > conf_thresh]
            boxes = nms(boxes, nms_thresh)
            boxes = np.stack(boxes)
            boxes_true = target.cpu().numpy().reshape(-1, 5)
            assert len(target) % 50 == 0, 'max_pboxes in image.py "fill_truth_detection" is different from 50'
            boxes_true = boxes_true[50*k:50*(k+1)]
            boxes_true = boxes_true[boxes_true.max(1) > 0, 1:5]
            out_boxes = np.array([imgpath, boxes_true, boxes])
            np.save(out_path + str(i*len(imgpath) + k), out_boxes)

            boxes_pred = boxes[:, :4].copy()
            scores = boxes[:, 4].copy()
            boxes_pred = boxes_pred[scores > 0.03]
            scores = scores[scores > 0.03]
            map.append(map_iou(boxes_true, boxes_pred, scores)) if len(scores) > 0 else None

    map = np.array(map).mean()
    print('Validation output saved at ' + out_path)
    print('The mAP IoU is: {}'.format(map))


if __name__ == '__main__':
    # python validate.py -c cfg/chexdet.cfg -w backup/15.pt -d cfg/chexdet.data --conf_threshold 0.001 -o data/out/ -b 1 --nms_threshold 0.01
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str,
                        default='cfg/sketch.data', help='data definition file, will validate over "valid" file')
    parser.add_argument('--config', '-c', type=str,
                        default='cfg/sketch.cfg', help='network configuration file')
    parser.add_argument('--weights', '-w', type=str, nargs='+',
                        default=['backup/15.pt'], help='weights')
    parser.add_argument('--conf_threshold', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--nms_threshold', type=float, default=0.4, help='nms threshold')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IOU threshold for metrics')
    parser.add_argument('--out_path', '-o', type=str,
                        help='path to write box predictions in the shape (num_batches, batch_size) where each of these'
                             ' contains img paths, gt bb and predicted bb')
    parser.add_argument('--batch_size', '-b', type=int, default=16)
    FLAGS, _ = parser.parse_known_args()

    main()
