import numpy as np
from PIL import Image
# =====================================================================
class channel_gri:
    def __init__(self, img_g, img_r, img_i, backsub=False):
        self.red   = img_i
        self.green = img_r
        self.blue  = img_g
        self.check_image_shapes()
        if backsub:
            self.subtract_background()

    def check_image_shapes(self):
        if (np.shape(self.red) != np.shape(self.green)) or \
            (np.shape(self.red) != np.shape(self.blue)):
            raise "Image arrays are of different shapes, exiting"
            return False
        else:
            self.NX, self.NY = np.shape(self.red)

    def subtract_background(self):
        self.red   -= np.median(self.red)
        self.green -= np.median(self.green)
        self.blue  -= np.median(self.blue)

    def apply_scale(self, scales):
        assert len(scales) == 3
        s1,s2,s3 = scales
        mean = (s1 + s2 + s3)/3.0
        self.red   *= (s1/mean)
        self.green *= (s2/mean)
        self.blue  *= (s3/mean)

    def pjm_offset(self, offset):
        self.red   += offset
        self.green += offset
        self.blue  += offset

    def pjm_mask(self,threshold):
        tiny = 1e-12
        mask = self.red*0.0 + 1.0
        for image in (self.red, self.green, self.blue):
            image[np.isnan(image)] = 0.0
            image[np.isinf(image)] = 0.0
            mask[image < threshold] = 0.0
            mask[(image > -tiny) & (image < tiny)] = 0.0
        self.red   *= mask
        self.green *= mask
        self.blue  *= mask

    def lupton_stretch(self, Q, alpha, itype='sum'):
        if itype == 'sum':
            I = (self.red+self.green+self.blue) + 1e-10
        elif itype == 'rms':
            I = np.sqrt(self.red**2.0+self.green**2.0+self.blue**2.0) + 1e-10
        stretch = np.arcsinh(alpha*Q*I) / (Q*I)
        self.red   *= stretch
        self.green *= stretch
        self.blue  *= stretch

    def lupton_saturate(self,threshold=1.0):
        x = np.dstack((self.red, self.green,self.blue))
        maxpix = np.max(x, axis=-1)
        maxpix[maxpix<threshold] = 1.0
        self.red   /= maxpix
        self.green /= maxpix
        self.blue  /= maxpix

    def pack_up(self):
        x = np.zeros([self.NX,self.NY,3])
        x[:,:,0] = np.flipud(self.red)
        x[:,:,1] = np.flipud(self.green)
        x[:,:,2] = np.flipud(self.blue)
        x = np.clip(x,0.0,1.0)
        x = x*255
        return Image.fromarray(x.astype(np.uint8))

# ======================================================================
def compose_imgs(img_g, img_r, img_i, \
                 scales=(1.0,1.0,1.0), \
                 Q=1.0, alpha=1.0, saturation='color', \
                 masklevel=None, offset=None, \
                 outfile='color.png'):
    # -------------------------------------------------------------------
    object_gri = channel_gri(img_g, img_r, img_i)
    object_gri.apply_scale(scales)
    object_gri.lupton_stretch(Q, alpha, itype='rms')
    # -------------------------------------------------------------------
    if masklevel is not None:
        object_gri.pjm_mask(masklevel)
    if offset is not None:
        object_gri.pjm_offset(offset)
    # -------------------------------------------------------------------
    if saturation == 'color':
        object_gri.lupton_saturate(threshold=1.0)
    # -------------------------------------------------------------------
    image = object_gri.pack_up()
    image.save(outfile)
    # -------------------------------------------------------------------
    return image
