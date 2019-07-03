# coding: utf8
"""
由于登陆yy后，会收im消息，导致首页继续更新，请不要登陆yy.
"""
from functools import wraps
import math
import cv2
from PIL import Image
import Levenshtein

# -------------------以下参数需要执行脚本前根据手机，本地电脑实际情况进行调整

class BWImageCompare(object):
    """Compares two images (b/w)."""

    _pixel = 255
    _colour = False

    def __init__(self, imga, imgb, maxsize=64):
        """Save a copy of the image objects."""

        sizea, sizeb = imga.size, imgb.size

        newx = min(sizea[0], sizeb[0], maxsize)
        newy = min(sizea[1], sizeb[1], maxsize)

        # Rescale to a common size:
        imga = imga.resize((newx, newy), Image.BICUBIC)
        imgb = imgb.resize((newx, newy), Image.BICUBIC)

        if not self._colour:
            # Store the images in B/W Int format
            imga = imga.convert('I')
            imgb = imgb.convert('I')

        self._imga = imga
        self._imgb = imgb

        # Store the common image size
        self.x, self.y = newx, newy

    def _img_int(self, img):
        """Convert an image to a list of pixels."""

        x, y = img.size

        for i in range(x):
            for j in range(y):
                yield img.getpixel((i, j))

    @property
    def imga_int(self):
        """Return a tuple representing the first image."""

        if not hasattr(self, '_imga_int'):
            self._imga_int = tuple(self._img_int(self._imga))

        return self._imga_int

    @property
    def imgb_int(self):
        """Return a tuple representing the second image."""

        if not hasattr(self, '_imgb_int'):
            self._imgb_int = tuple(self._img_int(self._imgb))

        return self._imgb_int

    @property
    def mse(self):
        """Return the mean square error between the two images."""

        if not hasattr(self, '_mse'):
            tmp = sum((a - b) ** 2 for a, b in zip(self.imga_int, self.imgb_int))
            self._mse = float(tmp) / self.x / self.y

        return self._mse

    @property
    def psnr(self):
        """Calculate the peak signal-to-noise ratio."""

        if not hasattr(self, '_psnr'):
            self._psnr = 20 * math.log(self._pixel / math.sqrt(self.mse), 10)

        return self._psnr

    @property
    def nrmsd(self):
        """Calculate the normalized root mean square deviation."""

        if not hasattr(self, '_nrmsd'):
            self._nrmsd = math.sqrt(self.mse) / self._pixel

        return self._nrmsd

    @property
    def levenshtein(self):
        """Calculate the Levenshtein distance."""

        if not hasattr(self, '_lv'):
            stra = ''.join((chr(x) for x in self.imga_int))
            strb = ''.join((chr(x) for x in self.imgb_int))

            lv = Levenshtein.distance(stra, strb)

            self._lv = float(lv) / self.x / self.y

        return self._lv


class ImageCompare(BWImageCompare):
    """Compares two images (colour)."""

    _pixel = 255 ** 3
    _colour = True

    def _img_int(self, img):
        """Convert an image to a list of pixels."""

        x, y = img.size

        for i in range(x):
            for j in range(y):
                pixel = img.getpixel((i, j))
                yield pixel[0] | (pixel[1] << 8) | (pixel[2] << 16)

    @property
    def levenshtein(self):
        """Calculate the Levenshtein distance."""

        if not hasattr(self, '_lv'):
            stra_r = ''.join((chr(x >> 16) for x in self.imga_int))
            strb_r = ''.join((chr(x >> 16) for x in self.imgb_int))
            lv_r = Levenshtein.distance(stra_r, strb_r)

            stra_g = ''.join((chr((x >> 8) & 0xff) for x in self.imga_int))
            strb_g = ''.join((chr((x >> 8) & 0xff) for x in self.imgb_int))
            lv_g = Levenshtein.distance(stra_g, strb_g)

            stra_b = ''.join((chr(x & 0xff) for x in self.imga_int))
            strb_b = ''.join((chr(x & 0xff) for x in self.imgb_int))
            lv_b = Levenshtein.distance(stra_b, strb_b)

            self._lv = (lv_r + lv_g + lv_b) / 3. / self.x / self.y

        return self._lv


class FuzzyImageCompare(object):
    """Compares two images based on the previous comparison values."""

    def __init__(self, imga, imgb, lb=1, tol=15):
        """Store the images in the instance."""

        self._imga, self._imgb, self._lb, self._tol = imga, imgb, lb, tol

    def compare(self):
        """Run all the comparisons."""

        if hasattr(self, '_compare'):
            return self._compare

        lb, i = self._lb, 2

        diffs = {
            'levenshtein': [],
            'nrmsd': [],
            'psnr': [],
        }

        stop = {
            'levenshtein': False,
            'nrmsd': False,
            'psnr': False,
        }

        while not all(stop.values()):
            cmp = ImageCompare(self._imga, self._imgb, i)

            diff = diffs['levenshtein']
            if len(diff) >= lb + 2 and \
                            abs(diff[-1] - diff[-lb - 1]) <= abs(diff[-lb - 1] - diff[-lb - 2]):
                stop['levenshtein'] = True
            else:
                diff.append(cmp.levenshtein)

            diff = diffs['nrmsd']
            if len(diff) >= lb + 2 and \
                            abs(diff[-1] - diff[-lb - 1]) <= abs(diff[-lb - 1] - diff[-lb - 2]):
                stop['nrmsd'] = True
            else:
                diff.append(cmp.nrmsd)

            diff = diffs['psnr']
            if len(diff) >= lb + 2 and \
                            abs(diff[-1] - diff[-lb - 1]) <= abs(diff[-lb - 1] - diff[-lb - 2]):
                stop['psnr'] = True
            else:
                try:
                    diff.append(cmp.psnr)
                except ZeroDivisionError:
                    diff.append(-1)  # to indicate that the images are identical

            i *= 2  
            if i==128:
                break;

        self._compare = {
            'levenshtein': 100 - diffs['levenshtein'][-1] * 100,
            'nrmsd': 100 - diffs['nrmsd'][-1] * 100,
            'psnr': diffs['psnr'][-1] == -1 and 100.0 or diffs['psnr'][-1],
        }

        return self._compare

    def similarity(self):
        """Try to calculate the image similarity."""

        cmp = self.compare()
        lnrmsd = (cmp['levenshtein'] + cmp['nrmsd']) / 2
        return lnrmsd
        return min(lnrmsd * cmp['psnr'] / self._tol, 100.0)  # TODO: fix psnr!


#return 图片相似度
def compare_images(image1, image2):
    images = {1: image1, 2: image2}

    results, i = {}, 1
    for namea, imga in images.items():
        for nameb, imgb in images.items():
            if namea == nameb or (nameb, namea) in results:
                continue
            cmp = FuzzyImageCompare(imga, imgb)
            sim = cmp.similarity()
            results[(namea, nameb)] = sim
            i += 1

    res = max(results.values())
    return res

if __name__ == "__main__":
    i = 0
    k = 1059
    box = (350,650,730,980)
    image1 = Image.open('/Users/test/yy/android/splash/test/{0}.jpg'.format(k))
    image1 = image1.crop(box)
    for i in range(k+1,k+10):
        image2 = Image.open('/Users/test/yy/android/splash/test/{0}.jpg'.format(i))
        image2 = image2.crop(box)
    #image3 = Image.open('/Users/test/Downloads/quality_game_6M_1080p_3000359.bmp')
        print 'index ',i,' ', compare_images(image1,image2)
        image1 = image2
   # # print 'quality_game_6M_1080p_3000301 ', compare_images(image1,image3)

    
