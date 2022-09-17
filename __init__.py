# -*- coding: utf-8 -*-

__version__ = '0.1'
__author__ = 'Shin Asakawa'
__email__ = 'asakawa@ieee.org'
__license__ = 'MIT'
__copyright__ = 'Copyright 2022 {0}'.format(__author__)

#from .bit import bit
from .bit import BIT
from .bit import BIT_LineBisection
from .bit import get_object_detection_model
from .bit import plot_img_bbox
from .a_utils import *
from .torch_nikogamulin_resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .torch_LeNet_Imagenet import LeNet_Imagenet

from .noto_fonts import *
from .noto_fonts import get_notoen_fonts

fonts_en = get_notoen_fonts(verbose=False)
fonts_ja = get_notojp_fonts(verbose=False)

from .noto_fonts import notoen_dataset
from .noto_fonts import get_notoen_fonts
from .noto_fonts import notojp_dataset
from .noto_fonts import get_notojp_fonts
from .torch_train_model import train_model
from .torch_MLP_Imagenet import MLP_Imagenet
from .pil_get_text_img import get_text_img
