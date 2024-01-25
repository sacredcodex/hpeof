import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, save_path):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    #cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    #cv2.waitKey()
    save_path = 'raft_result/'+save_path
    cv2.imwrite(save_path, (img_flo * 255).astype(np.uint8))


def get_subdirectories(folder_path):
    subdirectories = []
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            subdirectories.append(os.path.join(root, dir_name))
    return subdirectories


def demo(args, path):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()



    with torch.no_grad():
        #images = glob.glob(os.path.join(args.path, '**', '*.png'), recursive=True) + \
        #         glob.glob(os.path.join(args.path, '**', '*.jpg'), recursive=True)
        images = glob.glob(os.path.join(path, '*.png')) + \
                 glob.glob(os.path.join(path, '*.jpg'))
        
        images = sorted(images)
        for i in range(len(images) - 1):
            imfile1, imfile2 = images[i], images[i + 1]
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            scale_factor = 0.25

            new_width = int(image1.shape[3] * scale_factor)
            new_height = int(image1.shape[2] * scale_factor)

            image1 = torch.nn.functional.interpolate(image1, size=(new_height, new_width), mode='bilinear',
                                                     align_corners=False)
            image2 = torch.nn.functional.interpolate(image2, size=(new_height, new_width), mode='bilinear',
                                                     align_corners=False)


            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

            output_file = f'{i}_{i+1}'

            np.save(os.path.join(path, f'{i}_{i+1}.npy'), flow_up.cpu().numpy())
            #print(f"Array saved to '{output_file}'")

            '''
            result_path = f'result_{i}.jpg'
            viz(image1, flow_up, result_path)
            print(f'Saved visualization result to {result_path}')
            '''

if __name__ == '__main__':
    print('start')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    subdirectories = get_subdirectories(args.path)
    for path in subdirectories:
        print(path)
        demo(args, path)
