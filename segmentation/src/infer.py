import argparse

import torch
from path import Path

from dataloader import DataLoaderImgFile
from eval import evaluate
from net import WordDetectorNet
from visualization import *



def myfunc(img_dir):
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()

    net = WordDetectorNet()
    net.load_state_dict(torch.load('segmentation/model/weights', map_location=torch.device('cpu')))
    net.eval()
    net.to(args.device)

    loader = DataLoaderImgFile(Path('segmentation/src/result/'+img_dir), net.input_size, args.device)
    res = evaluate(net, loader, max_aabbs=1000)

    for i, (img, aabbs) in enumerate(zip(res.batch_imgs, res.batch_aabbs)):
        f = loader.get_scale_factor(i)
        aabbs = [aabb.scale(1 / f, 1 / f) for aabb in aabbs]
        img = loader.get_original_img(i)
        visualize(img, aabbs,img_dir)
        # visualize_and_plot(img, aabbs)


#if __name__ == '__main__':
#    main()
