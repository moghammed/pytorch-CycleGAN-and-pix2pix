# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import argparse

# from options.test_options import TestOptions

from models.models import create_model
import util.util as util


def main():
    parser = argparse.ArgumentParser(description='Process some image')

    # options = TestOptions()
    parser.add_argument('name', type=str, help='model to use')
    parser.add_argument('input', type=str, help='path to input image')
    parser.add_argument('output', type=str, help='path to output image')
    parser.add_argument('size', type=int, help='input size')
    parser.add_argument('--model', type=str, default='test', help='type of model to use')
    parser.add_argument('--epoch', '-e', type=str, default='latest', help='epoch to use')

    # Parse argument options
    opt = parser.parse_args()

    

    # if(opt.model != 'cycle_gan'):
    opt.which_direction = 'AtoB'
    opt.which_model_netG = 'unet_'+(str(opt.size))
    # else:
    #     opt.which_model_netG = 'resnet_9blocks'
    opt.model = 'test'
    opt.gpu_ids = [0]
    opt.isTrain = 0
    opt.checkpoints_dir = ''
    opt.batchSize = 1
    opt.loadSize = opt.size
    opt.fineSize = opt.size
    opt.input_nc = 3
    opt.output_nc = 3
    opt.phase = 'test'
    opt.dataset_mode = 'single'

    # opt.align_data = True
    opt.ngf=64
    opt.ndf=64

    opt.phase = 'Test'

    opt.norm = 'instance'
    opt.use_dropout = False

    opt.which_epoch = opt.epoch

    opt.checkpoints_dir = '/pix2pix/checkpoints'

    # Some defaults values for generating purpose

    # Load model
    model = create_model(opt)

    # Load image
    real = Image.open(opt.input)
    preprocess = transforms.Compose([
        transforms.Scale(opt.loadSize),
        transforms.CenterCrop(opt.fineSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    # Load input
    input_A = preprocess(real).unsqueeze_(0)
    model.input_A.resize_(input_A.size()).copy_(input_A)
    # Forward (model.real_A) through G and produce output (model.fake_B)
    model.test()

    # Convert image to numpy array
    fake = util.tensor2im(model.fake_B.data)
    # Save image
    util.save_image(fake, opt.output)


if __name__ == '__main__':
    main()
