import argparse
import os
import shutil
import logging
import sys
import h5py
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from networks.net_factory import net_factory
from utils import ramps
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/Prostate', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='DCNet', help='experiment_name')
parser.add_argument('--model', type=str, default='mcnet_kd', help='model_na`me')
parser.add_argument('--num_classes', type=int,  default=2, help='output channel of network')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
parser.add_argument('--labeled_num', type=int, default=7, help='labeled data')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
args = parser.parse_args()

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, jc, hd95, asd

def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)


    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out_main = net(input)
            if len(out_main)>1:
                out_main=out_main[0]
            else:
                out_main=out_main
            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred

    if np.sum(prediction == 1)==0:
        first_metric = 0,0,0,0
    else:
        first_metric = calculate_metric_percase(prediction == 1, label == 1)

    if "Prostate" in FLAGS.root_path:
        second_metric = first_metric
        third_metric = first_metric
    else:
        if np.sum(prediction == 2)==0:
            second_metric = 0,0,0,0
        else:
            second_metric = calculate_metric_percase(prediction == 2, label == 2)
        if np.sum(prediction == 3)==0:
            third_metric = 0,0,0,0
        else:
            third_metric = calculate_metric_percase(prediction == 3, label == 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))


    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))

    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric



def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
    snapshot_path = "../model/Prostate_{}_{}_{}_labeled".format(FLAGS.model, FLAGS.exp, FLAGS.labeled_num)
    test_save_path = "../model/Prostate_{}_{}_{}_labeled/predictions/".format(FLAGS.model, FLAGS.exp, FLAGS.labeled_num)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    logging.basicConfig(filename=snapshot_path + "/detail.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    net = net_factory(net_type=FLAGS.model,in_chns=1, class_num=FLAGS.num_classes)
    save_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_model_path), strict=False)
    net = net.cuda()
    print("init weight from {}".format(save_model_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in image_list:
        first_metric, second_metric, third_metric = test_single_volume(case, net, test_save_path, FLAGS)
        first_metric = np.asarray(first_metric)
        second_metric = np.asarray(second_metric)
        third_metric = np.asarray(third_metric)
        single_avg_metric = (first_metric + second_metric + third_metric) / 3

        logging.info("%s : " % (case))
        logging.info('%s : dice : %f  jc : %f  hd95 : %f  asd : %f' % ("avg_metric", single_avg_metric[0], single_avg_metric[1], single_avg_metric[2], single_avg_metric[3]))
        logging.info('%s : dice : %f  jc : %f  hd95 : %f  asd : %f' % ("first_metric", first_metric[0], first_metric[1], first_metric[2], first_metric[3]))
        logging.info('%s : dice : %f  jc : %f  hd95 : %f  asd : %f' % ("second_metric", second_metric[0], second_metric[1], second_metric[2], second_metric[3]))
        logging.info('%s : dice : %f  jc : %f  hd95 : %f  asd : %f' % ("third_metric", third_metric[0], third_metric[1], third_metric[2], third_metric[3]))
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)

    avg_metric = [first_total / len(image_list), second_total / len(image_list), third_total / len(image_list)]
    return avg_metric, test_save_path


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    metric, test_save_path = Inference(FLAGS)
    print(metric)
    print((metric[0]+metric[1]+metric[2])/3)

    with open(test_save_path+'../performance.txt', 'w') as f:
        f.writelines('metric is {} \n'.format(metric))
        f.writelines('average metric is {}\n'.format((metric[0]+metric[1]+metric[2])/3))
