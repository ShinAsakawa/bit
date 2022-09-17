import os
import numpy as np
import requests
import zipfile
from glob import glob
import matplotlib.pyplot as plt
from termcolor import colored
from PIL import Image, ImageDraw, ImageFont

import torch
from torch.utils.data import Dataset
from torchvision import transforms


default_bgcolor = (255,255,255)           # 真っ白な背景
default_width, default_height = 224, 224  # 画像の縦横の大きさ
default_fontsize = 28                     # フォントサイズ
#default_fontsize = 56                     # フォントサイズ

def get_notojp_fonts(
    notofonts_dir:str='fonts',  # Noto フォントデータを保存するディレクトリ名
    fontsize:int=28,            # デフォルトフォントサイズ
    verbose=True,
    )->dict:

    NOTOfonts_urls=[
        'https://noto-website-2.storage.googleapis.com/pkgs/NotoSerifJP.zip',
        'https://noto-website-2.storage.googleapis.com/pkgs/NotoSansJP.zip',
    ]

    if not os.path.exists(notofonts_dir):
        os.mkdir(notofonts_dir)

    if verbose:
        print('Noto font の読み込み', end='...')
    for url in NOTOfonts_urls:
        zip_fname = url.split('/')[-1]
        if not os.path.exists(zip_fname):  # ファイルが存在しない場合，ダウンロードする
            print(f'url:{url}')
            r = requests.get(url)
            with open(zip_fname, 'wb') as f:
                total_length = int(r.headers.get('content-length'))
                print(f'{zip_fname} をダウンロード中 {total_length} バイト')
                f.write(r.content)

        # zip ファイルの解凍処理
        with zipfile.ZipFile(zip_fname, "r") as zip_fp:
            if verbose:
                print(f'{zip_fname} 解凍中', end="...")
            zip_fp.extractall(notofonts_dir)

    # 解凍した TrueType フォントのファイル名を取得
    notofonts_fnames = glob(os.path.join(notofonts_dir,'*otf'))

    notofonts = {}
    for font_fname in notofonts_fnames:
        #print(f'font_fname:{font_fname.split('/')[-1]}')
        font_name = font_fname.split('/')[-1].split('.')[0]
        #print(font_name)
        notofonts[font_name] = ImageFont.truetype(
            font=font_fname,
            size=fontsize)

    if verbose:
        print('\n読み込んだ Noto fonts の情報')
        for i, (k, v) in enumerate(sorted(notofonts.items())):
            print(f'{i:2d}',
                  colored(f'{k}', "blue", attrs=['bold']),
                  f' {v}')

    return notofonts


class notojp_dataset(Dataset):

    def __init__(
        self,
        font_dict:dict=None,           # フォント辞書 上の `noto_fonts` を仮定
        items:list=[c for c in '０１２３４５６７８９'],  # 文字集合
        color:[tuple or str] = 'black', # デフォルト前景色
        bgcolor:tuple=default_bgcolor,  # デフォルト背景色
        width:int=default_width,        # デフォルト刺激画面幅
        height:int=default_height,      # デフォルト刺激画面高さ
        fontsize:int=default_fontsize,  # デフォルトフォントサイズ
        reps:int=1,                     # データの重複回数
        transform=None,
        target_transform=None,
    ):

        super().__init__()
        self.items = items
        self.font_dict = font_dict
        self.width = width
        self.heigh = height

        x0 = (width >> 1) - (fontsize >> 1) # - 4
        y0 = (height >> 1) - (fontsize >> 1) - 10
        imgs, labels = [], []
        for itm in items:
            for font in font_dict.keys():
                img = Image.new(mode='RGB',
                                size=(width, height),
                                color=bgcolor)
                draw_canvas = ImageDraw.Draw(img)
                draw_canvas.text(
                    xy=(x0, y0),
                    text=itm,
                    font=font_dict[font], # ['data'],
                    stroke_width=1,
                    #stroke_fill="black",
                    #spacing=-4,
                    #fill=(0,0,0),
                    fill=color,
                )
                imgs.append(np.array(img))
                labels.append((items.index(itm),itm))
                #imgs.append(torch.Tensor(np.array(img).transpose(2,0,1)))
        self.imgs = imgs
        self.labels = labels

        # RGB 各チャンネルの平均と分散の定義。CNN 唯一の前処理
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=mean, std=std)


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        _img = torch.Tensor(img.transpose(2,0,1))
        _img = self.normalize(_img)
        return _img, label
        #return torch.Tensor(img.transpose(2,0,1)), label

    def __getoriginalitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        return img, label



def get_notoen_fonts(
    font_dir:str = 'bit/', # os.path.join(HOME, 'Downloads'),
    font_zipfiles:list=['Noto_Serif.zip', 'Noto_Sans.zip'],
    verbose=True):
                     
    fonts = {}
    for font_zipfile in font_zipfiles:
        file_path = os.path.join(os.path.join(font_dir, font_zipfile))
        with zipfile.ZipFile(file_path, 'r') as zip_fp:
            with zipfile.ZipFile(file_path, 'r') as zip_fp_:
                for i, x in enumerate(zip_fp_.infolist()):
                    if 'ttf' in x.filename:
                        fonts[x.filename.split('.')[0]] = ImageFont.truetype(zip_fp.open(x.filename), size=16)
 
    notofonts = fonts
    if verbose:
        print('\n読み込んだ Noto fonts の情報')
        for i, (k, v) in enumerate(sorted(notofonts.items())):
            print(f'{i:2d}',
                  colored(f'{k}', "blue", attrs=['bold']), f' {v}')
                        
                        
    return fonts

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class notoen_dataset(Dataset):

    def __init__(
        self,
        fonts_dict:dict=None,           # フォント辞書 Noto font を仮定
        #fonts_dict:dict=notofonts,      # フォント辞書 Noto font を仮定
        items:list=[c for c in '0123456789'],  # 文字集合
        color:[tuple or str] = 'black', # デフォルト前景色
        bgcolor:tuple=default_bgcolor,  # デフォルト背景色
        width:int=default_width,        # デフォルト刺激画面幅
        height:int=default_height,      # デフォルト刺激画面高さ
        fontsize:int=default_fontsize,  # デフォルトフォントサイズ
        reps:int=1,                     # データの重複回数
        transform=None,
        target_transform=None):

        super().__init__()
        self.items = items
        self.fonts_dict = fonts_dict
        self.width = width
        self.heigh = height

        x0 = (width >> 1) - (fontsize >> 1) # - 4
        y0 = (height >> 1) - (fontsize >> 1) - 10
        imgs, labels = [], []
        for itm in items:
            for font in fonts_dict.keys():
                img = Image.new(mode='RGB',
                                size=(width, height),
                                color=bgcolor)
                draw_canvas = ImageDraw.Draw(img)
                draw_canvas.text(
                    xy=(x0, y0),
                    text=itm,
                    font=fonts_dict[font], # ['data'],
                    stroke_width=1,
                    #stroke_fill="black",
                    #spacing=-4,
                    #fill=(0,0,0),
                    fill=color,
                )
                imgs.append(np.array(img))
                labels.append((items.index(itm),itm))
                #imgs.append(torch.Tensor(np.array(img).transpose(2,0,1)))
        self.imgs = imgs
        self.labels = labels

        # RGB 各チャンネルの平均と分散の定義。CNN 唯一の前処理
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=mean, std=std)


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        _img = torch.Tensor(img.transpose(2,0,1))
        _img = self.normalize(_img)
        return _img, label
        #return torch.Tensor(img.transpose(2,0,1)), label

    def __getoriginalitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        return img, label

