import openslide
from utils.sdpc_python import sdpc
import numpy as np
import cv2

class slideCroper:

    def __init__(self, path, filename, level=0):
        self.path = path
        self.filename = filename
        self.level = level

    def get_wh(self):
        tail = self.filename[self.filename.rfind('.'):]

        try:
            if tail == '.mrxs':
                slide = openslide.OpenSlide(self.path + self.filename)
                [w, h] = slide.dimensions        

            elif tail == '.svs':
                slide = openslide.open_slide(self.path + self.filename)
                [w, h] = slide.dimensions

            elif tail == '.sdpc':
                slide = sdpc.Sdpc()
                slide.open(self.path + self.filename)

                w = slide.getAttrs()['width']
                h = slide.getAttrs()['height']
            else:
                print('Type Error')
                exit()
                w = None
                h = None
        except:
            
            w = None
            h = None
        return w, h

    def crop(self, x, y, w, h):

        tail = self.filename[self.filename.rfind('.'):]
        try:
            if tail == '.mrxs':
                slide = openslide.OpenSlide(self.path + self.filename)
                region = np.array(slide.read_region((int(x), int(y)), self.level, (int(w), int(h))))
                region = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
            elif tail == '.svs':
                slide = openslide.open_slide(self.path + self.filename)
                region = np.array(slide.read_region((int(x), int(y)), self.level, (int(w), int(h))))
                region = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
            elif tail == '.sdpc':
                slide = sdpc.Sdpc()
                slide.open(self.path + self.filename)
                region = np.ctypeslib.as_array(slide.getTile(self.level, int(y), int(x), int(w), int(h))).reshape((int(h), int(w), 3))
                region.dtype = np.uint8
            else:
                print('Type Error')
                exit()
        except:
            region = None
        return region