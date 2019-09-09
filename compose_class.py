import numpy as np
from PIL import Image
# =====================================================================
class channel_RGB(object):
    def __init__(self, RED=None, GREEN=None, BLUE=None, backsub=False):
        self.red   = RED
        self.green = GREEN
        self.blue  = BLUE
        self.check_image_shapes()
        if backsub:
            self.subtract_background()

    def check_image_shapes(self):
        if (np.shape(self.red) != np.shape(self.green)) or \
            (np.shape(self.red) != np.shape(self.blue)):
            raise "Image arrays are of different shapes, exiting"
        else:
            self.NX, self.NY = np.shape(self.red)

    def subtract_background(self):
        self.red   -= np.median(self.red)
        self.green -= np.median(self.green)
        self.blue  -= np.median(self.blue)

    def apply_scale(self, scales=(1.0,1.0,1.0)):
        assert len(scales) == 3
        s1,s2,s3 = scales
        mean = (s1 + s2 + s3)/3.0
        self.red   *= (s1/mean)
        self.green *= (s2/mean)
        self.blue  *= (s3/mean)

    def pjm_offset(self, offset=0.0):
        if offset==None:
            pass
        else:
            self.red   += offset
            self.green += offset
            self.blue  += offset

    def pjm_mask(self,masklevel=None):
        if masklevel==None:
            pass
        else:
            tiny = 1e-12
            mask = self.red*0.0 + 1.0
            for image in (self.red, self.green, self.blue):
                image[np.isnan(image)] = 0.0
                image[np.isinf(image)] = 0.0
                mask[image < masklevel] = 0.0
                mask[(image > -tiny) & (image < tiny)] = 0.0
            self.red   *= mask
            self.green *= mask
            self.blue  *= mask

    def lupton_stretch(self, Q=1.0, alpha=1.0, itype='sum'):
        if itype == 'sum':
            I = (self.red+self.green+self.blue) + 1e-10
        elif itype == 'rms':
            I = np.sqrt(self.red**2.0+self.green**2.0+self.blue**2.0) + 1e-10
        stretch = np.arcsinh(alpha*Q*I) / (Q*I)
        self.red   *= stretch
        self.green *= stretch
        self.blue  *= stretch

    def lupton_saturate(self,threshold=1.0, saturation='white', unsat=0.995):
        if saturation=="white":
            pass
        elif saturation=="color":
            x = np.dstack((self.red, self.green,self.blue))
            maxpix = np.max(x, axis=-1)
            maxpix[maxpix<threshold] = 1.0
            self.red   /= maxpix
            self.green /= maxpix
            self.blue  /= maxpix
        else:
            print("Not a recognized type of saturation!!!")

        all_tmp = np.hstack([self.red.ravel(), self.green.ravel(), self.blue.ravel()])
        self.red    /= (all_tmp[all_tmp.argsort()[int(np.round(len(all_tmp)*unsat))]])
        self.green  /= (all_tmp[all_tmp.argsort()[int(np.round(len(all_tmp)*unsat))]])
        self.blue   /= (all_tmp[all_tmp.argsort()[int(np.round(len(all_tmp)*unsat))]])

    def pack_up(self, unsat=0.995):
        x = np.zeros([self.NX,self.NY,3])
        x[:,:,0] = np.flipud(self.red)
        x[:,:,1] = np.flipud(self.green)
        x[:,:,2] = np.flipud(self.blue)
        # x = x/(x.ravel()[x.ravel().argsort()[int(np.round(len(x.ravel())*unsat))]])
        x = np.clip(x,0.0,1.0)
        x = x*255
        self.imgRGB = Image.fromarray(x.astype(np.uint8))
