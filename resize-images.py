import PIL
from PIL import Image
from pathlib import Path

path_string = 'img/sigiriya'
path = Path(path_string)

for idx, imgName in enumerate(path.glob("*.jpg")):
    img = Image.open(imgName)
    img = img.resize((200, 200), PIL.Image.ANTIALIAS)
    img.save('sigiriya/' + str(idx)+'.jpg')