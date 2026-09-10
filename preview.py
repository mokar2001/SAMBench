"""Scientific QA overlay: GT green, prediction magenta, correct overlap white."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from common import decode

p = argparse.ArgumentParser()
p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
p.add_argument('--run',type=Path,required=True)
p.add_argument('--image-id',type=int,required=True)
args = p.parse_args()
root = args.root.resolve()
gt = json.loads((root/f'prepared/instances/{args.image_id:04d}.json').read_text())
pred = json.loads((args.run/f'images/{args.image_id:04d}.json').read_text())
lookup = {r['instance_id']:r for r in pred['instances']}
tiles = []
for ann in gt['annotations']:
    result = lookup[ann['id']]
    im = np.array(Image.open(root/gt['image']['file_name']).convert('RGB'))
    g,pr = decode(ann['segmentation']),decode(result['segmentation'])
    for selected,color in [(g & ~pr,[50,220,90]),(pr & ~g,[235,50,200]),(g & pr,[235,235,235])]:
        im[selected] = (im[selected]*0.45+np.array(color)*0.55).astype(np.uint8)
    tile = Image.fromarray(im)
    draw = ImageDraw.Draw(tile)
    draw.rectangle(result['prompt_box_xyxy'],outline=(255,180,30),width=2)
    x1,y1,x2,y2=ann['box_xyxy']
    tile=tile.crop((max(0,x1-35),max(0,y1-35),min(tile.width,x2+35),min(tile.height,y2+35)))
    tile.thumbnail((210,240))
    canvas=Image.new('RGB',(230,280),(22,26,32))
    canvas.paste(tile,((230-tile.width)//2,30))
    ImageDraw.Draw(canvas).text((8,8),f'Tooth {ann["category_id"]} | Dice {result["dice"]:.3f}',fill='white')
    tiles.append(canvas)
sheet=Image.new('RGB',(230*8,280*((len(tiles)+7)//8)+40),(22,26,32))
ImageDraw.Draw(sheet).text((10,12),'GT only: green | Prediction only: magenta | Overlap: white | Prompt box: orange',fill='white')
for i,tile in enumerate(tiles):
    sheet.paste(tile,((i%8)*230,(i//8)*280+40))
dest=args.run/f'preview_{args.image_id:04d}.png'
sheet.save(dest)
print(dest)
