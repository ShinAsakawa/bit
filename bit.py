"""
半側空間無視の検査として Beheviorul Inattention Test:BIT が挙げられる。
BIT は 1987 年に Barbara Wilson, Janet Cockburn and Peter Halligan によって開発された。

BIT 4 tasks のうち，line_bisection, letter_cancellation についての実装を試みた。
line_cancellation と star_cancellation は未着手である。

- 線分二等分テスト line bisection 
- 線分抹消テスト line cancellation
- 文字抹消テスト letter cancellation
- 星印抹消テスト star cancellation

- source: https://strokengine.ca/en/assessments/behavioral-inattention-test-bit/

* **線分抹消課題 line crossing**: ページ上のすべての線を検出して消すことが求められる。
検査者は中央の列にある 4 本の線のうち，2 本を消すことで課題の内容を患者に示し，次に患者にページ上に見えるすべての線を消すように指示する。
* **文字抹消課題 letter cancellation** : 紙と鉛筆を使用した検査で，患者は妨害刺激である文字の埋め込まれた画像から指定された標的文字を走査して探し出し，消去することが求められる。
一枚の刺激図版には 34個 の大文字が 5 列に並んでいる。
40 個の標的刺激は，ページの両側に同数ずつ配置されている。
各文字の高さは 6 mmで、2 mmの間隔で配置されています。
* **星印抹消課題 star cancellation**: 言語および非言語刺激のランダムな配列で構成される。
52 個の大きな星 (14mm)，ランダムに配置された 13 個の文字，19 個の短い (3-4文字) 単語が，56 個の小さな星 (8mm) の中に散りばめられており，この小さな星印が標的刺激となる。
患者は小さな星をすべて抹消するように指示される。
* 図と形の模写:  被検査者はページの左側に描かれた 3 つの単純な絵をコピーするよう求められる。
3 つの絵 (四角い星，立方体，デイジー) は縦に配置されており，患者に明確に示される。
* 2 番目のテストでは、別の刺激シートに提示された 3 つの幾何学的図形の群を模写することが求められる。
前の項目とは異なり，ページの内容は患者に指摘されない。
* **線分二等分課題 line bisection**:  患者は水平線の中点を推定して示すことを求められる。
左無視の患者は，真の中心よりも右側の中間点を選択することが予想される。
それぞれの患者には 8 インチの太さ 1 mmの黒い水平線が 3 本，階段状に表示されている。
それぞれの線の範囲は，患者にはっきりと指摘され，患者は中心をマークするように指示される。

<!-- * Line crossing: Patients are required to detect and cross out all target lines on a page. 
When administering the test, the examiner demonstrates the nature of the task to the patient by crossing out two of four lines located in a central column, and then instructing them to cross out all lines they can see on the page.
* Letter Cancellation: Paper and pencil test in which patients are required to scan, locate, and cross out designated targets from a background of distractor letters. 
The test consists of 5 rows of 34 upper case letters presented on a rectangular page. 
Forty target stimuli are positioned such that each appears in equal number on both sides of the page. 
Each letter is 6 mm high and positioned 2 mm apart.
* Star Cancellation: This test consists of a random array of verbal and non-verbal stimuli. 
The stimuli are 52 large stars (14 mm), 13 randomly positioned letters and 19 short (3-4 letters) words are interspersed with 56 smaller stars (8mm) which comprise the target stimuli. 
The patient is instructed to cancel all the small stars.
* Figure and Shape copying: In this test, the patient is required to copy three separate, simple drawings from the left side of the page. 
The three drawings (a four pointed star, a cube, and a daisy) are arranged vertically and are clearly indicated to the patient. 
* The second part of the test requires the patient to copy a group of three geometric shapes presented on a separate stimulus sheet. 
Unlike the previous items, the contents of the page are not pointed out to the patient.
* Line Bisection: Patients are required to estimate and indicate the midpoint of a horizontal line. 
The expectation is that the patient with left neglect will choose a midpoint to the right of true center. 
Each patient is presented with three horizontal, 8-inch black lines, 1- mm thick, displayed in a staircase fashion across the page. 
The extent of each line is clearly pointed out to the patient who is then instructed to mark the center.
-->
 
#### 文献

- Halligan, P., Cockburn, J., Wilson, B. (1991). The Behavioural Assessment of Visual Neglect. Neuropsychological Rehabilitation 1, 5-32.
- 石合純夫代表 日本版差規制委員会 (1999) BIT 行動性無視検査日本版，新興医学出版社，東京
"""

import os
import sys
import numpy as np
from glob import glob
import PIL
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import matplotlib.pyplot as plt

import torch

# Noto Truetype font の読み込み
# Download NotoSerif and NotoSans fonts from https://fonts.google.com/download?family=Noto%20Serif%20JP
#_ = ImageFont.truetype(os.path.join('/Users/asakawa/study/data/Noto_JP_fonts','NotoSerifJP-Bold.otf'))
from IPython import get_ipython
isColab =  'google.colab' in str(get_ipython())
if isColab:
    noto_font_dir = 'Noto_JP_fonts'
    bit_image_dir = '2022muto_figures'

    #!mkdir Noto_JP_fonts

    #!wget https://noto-website-2.storage.googleapis.com/pkgs/NotoSerifJP.zip
    #!wget https://noto-website-2.storage.googleapis.com/pkgs/NotoSansJP.zip
    #!unzip NotoSerifJP.zip -d Noto_JP_fonts
    #!unzip -o NotoSansJP.zip -d Noto_JP_fonts  # `-o` means overwrite 
else:
    noto_font_dir = '/Users/asakawa/study/data/Noto_JP_fonts/'
    bit_image_dir = '/Users/asakawa/study/2022muto/figures'
notofonts_fnames = glob(os.path.join(noto_font_dir,'*otf'))
#print(len(notofonts_fnames))
notofonts = {fname.split('/')[-1].split('.')[0]:{'fname':fname} for fname in notofonts_fnames}
for fontname in notofonts.keys():
    notofonts[fontname]['data'] = ImageFont.truetype(notofonts[fontname]['fname'])


class BIT():
    '''
    Behavioural Inattention Test 行動非注意試験とでも訳すのか，の実装
    4 つの課題を実装予定
        1. 線分二等分課題
        2. 線分消去課題
        3. 文字消去課題
        4. 星印消去課題

    - source: https://strokengine.ca/en/assessments/behavioral-inattention-test-bit/

    * 線分抹消課題 line crossing: ページ上のすべての線を検出して消すことが求められる。
        検査者は中央の列にある 4 本の線のうち，2 本を消すことで課題の内容を患者に示し，次に患者にページ上に見えるすべての線を消すように指示する。
    * 文字抹消課題 letter cancellation : 紙と鉛筆を使用した検査で，患者は妨害刺激である文字の埋め込まれた画像から指定された標的文字を走査して探し出し，消去することが求められる。
        一枚の刺激図版には 34個 の大文字が 5 列に並んでいる。
        40 個の標的刺激は，ページの両側に同数ずつ配置されている。
        各文字の高さは 6 mmで、2 mmの間隔で配置されています。
    * 星印抹消課題 star cancellation: 言語および非言語刺激のランダムな配列で構成される。
        52 個の大きな星 (14mm)，ランダムに配置された 13 個の文字，19 個の短い (3-4文字) 単語が，56 個の小さな星 (8mm) の中に散りばめられており，この小さな星印が標的刺激となる。
        患者は小さな星をすべて抹消するように指示される。
    * 線分二等分課題 line bisection:  患者は水平線の中点を推定して示すことを求められる。
        左無視の患者は，真の中心よりも右側の中間点を選択することが予想される。
        それぞれの患者には 8 インチの太さ 1 mmの黒い水平線が 3 本，階段状に表示されている。
        それぞれの線の範囲は，患者にはっきりと指摘され，患者は中心をマークするように指示される。

    * Line crossing: Patients are required to detect and cross out all target lines on a page. 
        When administering the test, the examiner demonstrates the nature of the task to the patient by crossing out two of four lines located in a central column, and then instructing them to cross out all lines they can see on the page.
    * Letter Cancellation: Paper and pencil test in which patients are required to scan, locate, and cross out designated targets from a background of distractor letters. 
        The test consists of 5 rows of 34 upper case letters presented on a rectangular page. 
        Forty target stimuli are positioned such that each appears in equal number on both sides of the page. 
        Each letter is 6 mm high and positioned 2 mm apart.
    * Star Cancellation: This test consists of a random array of verbal and non-verbal stimuli. 
        The stimuli are 52 large stars (14 mm), 13 randomly positioned letters and 19 short (3-4 letters) words are interspersed with 56 smaller stars (8mm) which comprise the target stimuli. 
        The patient is instructed to cancel all the small stars.
    * Line Bisection: Patients are required to estimate and indicate the midpoint of a horizontal line. 
        The expectation is that the patient with left neglect will choose a midpoint to the right of true center. 
        Each patient is presented with three horizontal, 8-inch black lines, 1- mm thick, displayed in a staircase fashion across the page. 
        The extent of each line is clearly pointed out to the patient who is then instructed to mark the center.
        
    note: 
    コード中 width:int=4662, height:int=3289, などの数値は武藤先生から送っていただいた図版のサイズの最小値
        
    Functions:
    - show_original_image(task): 各課題のスキャンした画像を表示する
        
    - draw_a_bisecition_line():
        
    - draw_a_bisection_line():
        
    - make_line_bisection_task_images():
        
    - draw_bbox(): バウンディングボックスを描画する
        
    - draw_text(): PIL.Image.Image データ内に文字データを書き込む
        
    - make_a_letter_cancellation_image():
        
    - make_letter_cancellation_task_images(N=10):

    '''
    
    def __init__(self, fontdata=notofonts, fontsize=128):
        
        self.fontdata = fontdata
        self.fontnames = [x for x in self.fontdata] # font 名のリスト
        self.default_font_name = 'NotoSansJP-Bold'                    # デフォルトフォントとする
        self.default_font = self.fontdata[self.default_font_name]['data'] # デフォルトフォント
        self.default_font_size = 128
            
        #武藤先生から送っていただいた BIT 日本語版の図版のスキャンデータを読み込む
        #bit_image_dir = '/Users/asakawa/study/2022muto/figures'
        bit_image_files = {'line_bisection': '20220124BIT_line_bisection.jpg',
                          'line_erasion': '20220124BIT_line_crossing.jpg',
                          'letter_erasion': '20220124BIT_letter_cancellation.jpg',
                          'star_erasion': '20220124BIT_star_cancellation.jpg'}

        #図版のスキャンデータと課題とを結びつける
        _tasks = {}
        for task, img_file in bit_image_files.items():
            fname = bit_image_files[task]  # bit_image_files 辞書からファイル名を得る
            img = Image.open(os.path.join(bit_image_dir, fname)).convert('RGB')
            _tasks[task] = {'fname': fname, 'img': img}
        self.tasks = _tasks
        
        self.max_width, self.max_height = 4662, 3289
        # 4 枚のスキャン画像の幅と高さの平均値を作成する刺激画像サイズとする
        #_width, _height = np.array(np.array([self.tasks[task]['img'].size for task in self.tasks]).mean(axis=0),dtype=int)
        #self.max_width, self.max_height = _width, _height

        self.colornames = list(PIL.ImageColor.colormap)

        self.hira_chars = " ".join(ch for ch in 'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん').split(' ')
        self.star = '★'
        self.line_symbol = '<line>'
        self.symbols = [self.line_symbol , self.star] + self.hira_chars


        # DETR のサンプルプログラムを借用
        COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
                  [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

        self.COLORS = COLORS * 100


         
    def make_a_canvas(self, 
                      width:int=None, 
                      height:int=None)->PIL.Image.Image:
        '''刺激を描くためのキャンバスを作成して返す'''
        width=self.max_width if width == None else width
        height=self.max_height if height == None else height
        print(f'width:{width}, height:{height}')
        img = Image.new(mode='RGB', 
                        size=(width, height), 
                        color=(255,255,255))  # 真っ白な (255,255,255) 画像
        draw = ImageDraw.Draw(img)  # draw オブジェクト ある種のキャンバスを生成
        return img, draw
    
    def set_new_canvas(self,
                      width:int=None, 
                      height:int=None)->None:
        self.canvas, self.draw = self.make_a_canvas()
        #return canvas, draw


    def set_font(self, 
                 fontname='NotoSansJP-Bold',
                 fontsize:int=None)->None:
        '''フォントを設定する'''

        #print(f'self.fonts:{self.fonts}')
        #sys.exit()
        #fontname = self.default_font if fontname == None else self.fonts[fontname]['data']
        self.fontsize = self.default_font_size if fontsize == None else 128
        #print(f'fontname:{fontname}')
        #print(self.fontdata[fontname])
        #print(f"self.fontdata[{fontname}]:{self.fontdata[fontname]}")
        #print(f'self.fontdata[fontname]["data"]:{self.fontdata[fontname]["data"]}')
        self.default_font = self.fontdata[fontname]['data']

        
    def show_original_image(self, 
                            task='line_bisection', 
                            figsize=(8,6), 
                            show_all=False):
        '''スキャンした図版の画像データを表示する'''
        if show_all:
            for task in self.tasks:
                plt.figure(figsize=figsize)        
                img = self.tasks[task]['img']
                plt.imshow(img)
                plt.show()
        else:
            plt.figure(figsize=figsize)
            img = self.tasks[task]['img']
            plt.imshow(img)
            plt.show()


    def get_original_imgs_torch(self):
        torch_imgs = []
        for task in self.tasks:
            img = self.tasks[task]['img']
            img = torch.from_numpy(np.array(img))   # 読み込んだ画像を PyTorch tensor へ変換
            img = img.narrow(1,0,4662)              # `star_cacellation.jpg` だけ横幅が 5 ドット少ないのでそれに合わせる
            torch_imgs.append(img.permute(2,0,1))   # PyTorch は channel first なので次元を変換
        return torch_imgs


    def dislay_self_canvas(self,
                figsize=(8,6),
                )->PIL.Image.Image:
        if not self.canvas:
            return
        else:
            plt.figure(figsize=figsize)
            plt.imshow(self.canvas)


    def draw_a_bisection_line(self,
							  img:PIL.Image.Image=None,
							  start_x:int=-1,
							  start_y:int=-1,
							  line_length:int=4000,
							  line_width:int=-1,
							  fg_color=None,
							  width:int=4662, 
                              height:int=3289)->PIL.Image.Image:
        '''線分二等分課題の線分 1 本を PIL.Image.Image データに書き込む'''
            
        if not isinstance(img, PIL.Image.Image):
            img = Image.new(mode='RGB', size=(width, height), color=(255,255,255))
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        else:
            img = img.copy()
        _width, _height = img.size         # PIL.Image.Image データの大きさを取得
        _half_width = _width >> 1
        draw = ImageDraw.Draw(img)         # draw キャンバスを生成

        x_offs = int(_width * 0.10)
        y_offs = int(_height * 0.10)        

        _start_x = x_offs                  # x 座標の開始点
        _start_y = y_offs                  # y 座標縦方向の開始点 左上が (0,0) であることに注意
        _max_width  = int(_width  * 0.80)  # x 座標の終端点
        _max_height = int(_height * 0.80)  # y 座標の終端点
        #_width = int(width * 0.90)        # x 座標の終端点
        #_height = int(height * 0.80)      # y 座標の終端点

        _line_length = int(_width * 0.7)   # 7 割で固定
        _start_x += np.random.randint(_max_width - _line_length) if start_x == -1 else start_x
        _start_y += np.random.randint((_max_height - _start_y) >> 3) if start_y == -1 else start_y
        #_line_length = (_half_width - _start_y) + np.random.randint(_max_width - _half_width)
        r,g,b = np.random.randint(256>>1,size=3) + 256 >> 2

        _line_width = np.random.randint(20) + 10 if line_width == -1 else line_width
        color = (r,g,b) if fg_color==None else fg_color
        #print(f'_start_x:{_start_x} _start_y:{_start_y} line_length:{line_length}, line_width:{line_width} color={color}')
        draw.line(xy=[(_start_x, _start_y), (_start_x + _line_length, _start_y)], width=_line_width, fill=color)

        # バウンディングボックス
        dx, dy = 60, 60
        xmin = _start_x - dx
        ymin = _start_y - dy - (_line_width>>1) 
        xmax = _start_x + _line_length + dx
        ymax = _start_y + (_line_width>>1) + dy
        bbox = [xmin,ymin,xmax,ymax]
        return img, bbox


    def pm_randint(self, _range:int)->int:
        """ プラスマイナス _range 幅の乱数を返す"""
        return np.random.randint(_range) - (_range >>1)
    
    def draw_bisection_lines(self,
                             img:PIL.Image.Image=None,
                             n_lines=3,
                             width:int=4662, height:int=3289)->PIL.Image.Image:
        '''線分二等分課題のための水平線を n_lines 本描画する'''
        if not isinstance(img, PIL.Image.Image):
            img = Image.new(mode='RGB', size=(width, height), color=(255,255,255))
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        else:
            img = img.copy()
        _width, _height = img.size
        #print(f'_width:{_width}, _height:{_height}')

        _start_y_offs = int(_height * 0.1)   # 線分描画 y 座標の開始点は 上から 10 % の地点とする
        _height_max   = int(_height * 0.9)   # 線分描画 y 座標の終了点は 上から 90 % の地点とする
        _line_offs = self.pm_randint(int(_width / (n_lines + 1)))
        _min_step_y = int( _width / (n_lines + 1)) >>2
        _start_y = _start_y_offs + _line_offs # np.random.randint(line_offs >> 1)


        bboxes = []       
        for _ in range(n_lines):
            img, bbox = self.draw_a_bisection_line(start_y=_start_y, img=img)
            _start_y = bbox[-1]  # bbox の最後の要素は，y 座標の最下点なので，そこを y の始点とする
            if _start_y >= _height_max:
                break
            _y_offs = self.pm_randint(int((_height_max - _start_y)/(n_lines+1)) )
            #_y_offs = np.random.randint(int((_height_max - _start_y)/(n_lines+1)))
            _start_y += _y_offs # + (height >> 4)
            #_start_y += _min_step_y + _y_offs # + (height >> 4)
            bboxes.append(bbox)
            
        return img, bboxes


    def make_line_bisection_task_images(self, 
                                        N=10, 
                                        n_lines=0,
                                        min_n_lines=3, 
                                        max_n_lines=7):
        '''線分二等分課題のための訓練データ画像を N 枚作成する'''
        images, bboxes = [], []
        for i in range(N):
            img = self.tasks['line_bisection']['img']
            if n_lines == 0:
                n_lines = np.random.randint(min_n_lines, max_n_lines)
            _img, _bbox = self.draw_bisection_lines(n_lines=n_lines)    
            images.append(_img)
            bboxes.append(_bbox)
            #images.append(self.draw_bisection_lines(n_lines=n_lines))
            
        return images, bboxes
    
    
    def draw_bbox(self,
                  fontname:str='NotoSansJP-Bold', 
                  fontsize:int= 128,
                  img:PIL.Image.Image=None,
                  x:int = -1, y:int = -1, 
                  fg_color:str='black', bg_color:str='white',
                  width:int=4662, 
                  height:int=3289,
                 )->PIL.Image.Image:
        if not isinstance(img, PIL.Image.Image):
            img = Image.new('RGB', (width, height), (255,255,255))
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        else:
            img = img.copy()
        width, height = img.size
        draw = ImageDraw.Draw(img)
        self.font = ImageFont.truetype(fontname, fontsize)
        
        top, bottom = int( 0.25 * height), int(0.70 * height)
        left, right = int(0.05 * width), int(0.95 * width)
        center_x, center_y = width // 2, height // 2

        #x0 = x if x != -1 else width // 2
        #y0 = y if y != -1 else height // 2
        #draw.text((x0, y0), text=text, fill=fg_color, font=font)
        draw.rectangle((left,top,right,bottom),fill=None, outline=fg_color, width=40)
        return img
    

    def write_text_on_self_canvas(self,
                                  text:str='はー',
                                  fontname:str='NotoSerifJP-Black',
                                  fontsize:int=128,
                                  fg_color:str="black",
                                  bg_color:str='white',
                                  width:int=4662, height:int=3289,
                                  x:int=-1,
                                  y:int=-1,
                                  bbox:bool=False)->PIL.Image.Image:
        if not self.canvas:
            self.set_new_canvas()
        if fg_color == 'rand':
            fg_color = np.random.choice(self.colornames)

        # fontname に従ってフォントを指定する 
        __font = ImageFont.truetype(self.fontdata[fontname]['fname'], size=fontsize)

        # 描画位置が指定されていなければ，ど真ん中に設定する
        x0 = x if x != -1 else width // 2
        y0 = y if y != -1 else height // 2

        # 文字サイズの取得
        size = self.draw.multiline_textsize(text=text, font=__font)
        #print(f'size:{size}')
        if x == -1:
            x0 -= size[0] >> 1
        if y == -1:
            y0 -= size[1] >> 1
        # 実際の `text` 内容を描画
        self.draw.text((x0, y0), text=text, fill=fg_color, font=__font)

        # 後処理
        size = self.draw.textbbox(xy=(x0,y0), text=text, font=__font)
        if bbox:
            self.draw.rectangle(xy=size, fill=None, outline=fg_color, width=15)

        
    def draw_text(self,
                  text:str='おはよう',
                  fontname:str='NotoSansJP-Bold', 
                  fontsize:int= 128,
                  img:PIL.Image.Image=None,
                  x:int = -1, y:int = -1, 
                  fg_color:str='black', 
                  bg_color:str='white',
                  width:int=4662, 
                  height:int=3289,
                  bbox=False)->PIL.Image.Image:
        
        if not isinstance(img, PIL.Image.Image):
            img = Image.new('RGB', (width, height), (255,255,255))
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        else:
            img = img.copy()
        width, height = img.size
        draw = ImageDraw.Draw(img)

        if fg_color == 'rand':
            fg_color = np.randcom.choice(self.colornames)

        # _old_font = draw.getfont()
        # if _old_font != self.font:
        #     self.set_font(fontname)
        #     #font = self.fonts[fontname]['data']
        #     #font = ImageFont.truetype(self.fontdata[fontname]['data'], fontsize)
        #     #self.font = ImageFont.truetype(self.fontdata[fontname]['fname'], fontsize)

        self.font = ImageFont.truetype(self.fontdata[fontname]['fname'], fontsize)

        # #self.set_font(fontname)
        # #self.font = ImageFont.truetype(self.fontdata[fontname]['fname'], fontsize)
        
        top, bottom = int(0.25 * height), int(0.70 * height)
        left, right = int(0.05 * width),  int(0.95 * width)
        center_x, center_y = width // 2, height // 2

        # 描画位置が指定されいなければ，ど真ん中に設定する
        x0 = x if x != -1 else width // 2
        y0 = y if y != -1 else height // 2

        # fontname に従ってフォントを指定する 
        __font = ImageFont.truetype(self.fontdata[fontname]['fname'], size=fontsize)

        # 実際の `text` 内容を描画
        draw.text((x0, y0), text=text, fill=fg_color, font=__font )

        # 後処理
        size = draw.textbbox(xy=(x0,y0), text=text, font=__font)
        if bbox:
            draw.rectangle(xy=size,fill=None, outline=fg_color, width=15)
        return img

    
    def make_a_letter_cancellation_image(self,
                                         img:PIL.Image.Image=None,
                                         fontname:str='NotoSansJP-Bold', 
                                         chars_per_line=34,
					                     fontsize:int=128,
                                         n_lines=5,
                                         width:int=4662, 
                                         height:int=3289,
                                         fg_color:str='black',
                                        )->PIL.Image.Image:
        '''文字抹消課題の画像データを一枚作成'''

        # 引数 img が指定されていなければ，新しい img 実体を作成        
        if not isinstance(img, PIL.Image.Image):
            img = Image.new('RGB', (width, height), (255,255,255))
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        #else:
        #    img = img.copy()
        width, height = img.size
        draw = ImageDraw.Draw(img)
        _old_font = draw.getfont()


        # 文字描画領域の指定
        _width  = int(0.90 * width)
        _height = int(0.95 * height)
        line_start_x = int(width  * 0.095)
        line_start_y = int(height * 0.310)
        #chars_per_line = chars_per_line
        font_size = _width // int(chars_per_line * 1.1)
        font_offset = font_size + (font_size >> 6)
        line_offset = int(font_size * 2.0)

        center_x = width >> 1
        bottom_y = int(height * 0.78)
        target_offset = 2 * font_offset

        # 引数 `fg_color` が指定されていなければ，乱数を用いて色を選択する
        if fg_color == 'rand':
            fg_color = np.random.choice(self.colornames)
            #print(f'fg_color:{fg_color}')

        target_line_x = center_x - target_offset
        x = center_x - target_offset
        y = bottom_y
        ret = self.draw_text(text='星', x=x, y=y, 
                             fg_color=fg_color, 
                             img=img, 
                             fontname=fontname,
                             )

        x += target_offset * 2
        y = bottom_y
        ret = self.draw_text(text='↑',x=x, y=y, fg_color='red', img=ret) # , font=self.default_font, fontsize=self.default_font_size)

        # 最低行に 矢印記号を描画
        up_arrow = '↑'
        x = center_x - (font_size > 1)
        y = height - (font_size * 2)
        ret = self.draw_text(text=up_arrow, x=x, y=y, fg_color='black', img=ret) # , font=self.default_font, fontsize=self.default_font_size)

        x0, y0 = line_start_x, line_start_y
        x, y = x0, y0
        for i in range(chars_per_line * n_lines):
            ret = self.draw_text(text=self.hira_sample()[0], 
                                 x=x, y=y, 
                                 fontname=fontname, 
                                 bbox=False, 
                                 img=ret, 
                                 fg_color=fg_color) # , font=font, fontsize=font_size)
            #ret = self.draw_text(text=self.hira_sample()[0], x=x, y=y, fontname=fontname, bbox=False, img=ret, fg_color='cyan') # , font=font, fontsize=font_size)
            x += font_offset
            if (i+1) % 34 == 0:
                y += line_offset
                x = x0
        return ret


    def make_letter_cancellation_task_images(self, N=10):
        """文字抹消課題のための訓練データ画像を N 枚作成する"""
        
        images = []
        for i in range(N):
            img = self.tasks['letter_eraseion']['img']
            images.append(self.make_a_letter_erasion_image())
            
        return images


    def hira_select(self, 
                    n=2,
                    verbose=False):
        """ひらがな文字を選んで，ターゲットと妨害刺激として設定する"""
        #if not self.hira_chars in locals():
        #    #if not isinstance(self.hira_chars):

        # 文字列のシャッフル
        self.hira_chars = np.random.permutation(self.hira_chars)

        self.target_hira_chars = self.hira_chars[:n]        # 最初の n 文字をターゲットに設定
        self.distractors_hira_chars =  self.hira_chars[n:]  # 残りの文字を 妨害刺激用文字として設定

        if verbose:
            print(f'self.target_hira_chars:{self.target_hira_chars}')
            print(f'self.distractors_hira_chars:{self.distractors_hira_chars}') 


    @staticmethod
    def hira_sample(n=2):
        s_hira = " ".join(ch for ch in 'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん').split(' ')
        a = np.random.permutation(s_hira)
        return a[:n]
            

if __name__ == "__main__":
	bit = BIT()	
	bit.show_original_image(show_all=True)
