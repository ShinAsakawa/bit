import PIL
from PIL import Image, ImageDraw, ImageFont
from . import fonts_en

default_width=224
default_height=224
default_bgcolor=(255,255,255)
default_fontsize=28

def get_text_img(
    text:str="XYZ",
    width:int=default_width,                  # デフォルト刺激画面幅
    height:int=default_height,                # デフォルト刺激画面高さ
    x0:int=0,                                 # 描画開始点左端
    y0:int=0,                                 # 描画開始点上端
    bgcolor:tuple=default_bgcolor,            # デフォルト背景色
    color:[tuple or str] = 'black',           # デフォルト前景色
    fontsize:int=default_fontsize,            # デフォルトフォントサイズ
    anchor:str='lt',                          # デフォルトアンカー 'lt' は左上の意
    draw_bbox:bool=False,                     # bbox を描画するか否か
    bbox_color:str='cyan',                    # bbox を描画する色
    bbox_width:int=2,                         # bbox を描画する線幅
    font:PIL.ImageFont.FreeTypeFont=fonts_en['NotoSans-Regular'],  # フォント
    target_transform=None):
    """引数 text で指定された 1 行の文字列を描画し，その PIL.Image.Image と PIL.ImageDraw.ImageDraw
    を返す。
    引数として `draw_bbox=True` が指定された場合にはバウンディングボックスを描画し，その座標を返す。
    """
    #anchor: テキストアンカーの位置決め。テキストに対するアンカーの相対的な位置を定める。
    #デフォルト位置は左上。
    #https://pillow.readthedocs.io/en/stable/handbook/text-anchors.html#text-anchors
    #このパラメータは TrueType フォント以外では無視される。
    #デフォルトの配置は左上で，具体的には水平方向のテキストは la (left-ascender) で，垂直方向のテキストは lt (left-top) である。
    #アンカーは 2 文字の文字列で指定する。
    #1 文字目は水平方向の配置を，2 文字目は垂直方向の配置を指す。
    #例えば，横書きテキストのデフォルト値である la は，左肩テキストを意味する。
    #`PIL.ImageDraw.ImageDraw.text()` でアンカーを指定してテキストを描画する場合,，指定したアンカー点が xy 座標になるようにテキストが配置される。
    #例えば，以下の画像では，テキストは ms (中基線 middle-baseline) 揃えで xy が 2 本の線の交点になるように配置される。

    
    img = Image.new(mode='RGB',
                    size=(width, height), 
                    color=bgcolor)
    draw_canvas = ImageDraw.Draw(img)
    
    # 画像のど真ん中に描画するため，描画する文字列のサイズを得る
    bbox = draw_canvas.textbbox(xy=(x0,y0),
                                font=font, 
                                text=text,
                                anchor=anchor)
    bbox_width  = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]

    if x0 == 0:  # 描画する文字列の左端を計算
        x0 = (width >> 1) - (bbox_width >> 1)
    if y0 == 0:  # 描画する文字列の上端を計算 
        y0 = (height >> 1) - (bbox_height >> 1)
    
    draw_canvas.text(xy=(x0, y0),   # 実際の描画
                     text=text,
                     font=font,
                     stroke_width=1,
                     #stroke_fill="black",
                     #spacing=-4,
                     #fill=(0,0,0),
                     anchor=anchor,
                     fill=color)
    
    if draw_bbox:
        bbox = draw_canvas.textbbox(xy=(x0,y0),
                                    font=font, 
                                    text=text,
                                    anchor=anchor)

        draw_canvas.rectangle(bbox,
                              outline=bbox_color,
                              fill=None,
                              width=2)
        return img, draw_canvas, bbox
    else:
        return img, draw_canvas
