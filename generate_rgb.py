import os
import sys
import numpy as np
import astropy.io.fits as pyfits
from compose_class import channel_RGB
# import aplpy


# def generate_rgb(src_path, dst_path, number=20000):
#     start = 100000
#     bands = ['Band1', 'Band2', 'Band3', 'Band4']
#     band_letter = ['R', 'I', 'G', 'U']
#     for i in range(number):
#         image_src = []
#         for j in range(3):
#             image = f'{bands[j]}/imageSDSS_{band_letter[j]}-{start + i}.fits'
#             image = os.path.join(src_path, image)
#             image_src.append(image)
#         temp_image = f'/tmp/temp_cube_{start + i}.fits'
#         temp_2d_image = f'/tmp/temp_cube_{start + i}_2d.fits'
#         dst_image = f'ground_based_{start + i}.png'
#         dst_image = os.path.join(dst_path, dst_image)
#         aplpy.make_rgb_cube(image_src, temp_image)
#         aplpy.make_rgb_image(temp_image, dst_image)
#         os.remove(temp_image)
#         os.remove(temp_2d_image)

def img_preproc(ifile, rfile, gfile):
    img_g = pyfits.getdata(gfile)
    img_g[np.where(img_g == 100.0)] = 0.0

    img_r = pyfits.getdata(rfile)
    img_r[np.where(img_r == 100.0)] = 0.0

    img_i = pyfits.getdata(ifile)
    img_i[np.where(img_i == 100.0)] = 0.0

    img_rscl = 1e-11
    img_thds = 1e-11

    img_g_rscl = img_g/img_rscl
    img_r_rscl = img_r/img_rscl
    img_i_rscl = img_i/img_rscl

    img_g_rscl[np.where(img_g_rscl <= img_thds)] = img_thds
    img_r_rscl[np.where(img_r_rscl <= img_thds)] = img_thds
    img_i_rscl[np.where(img_i_rscl <= img_thds)] = img_thds

    return img_g_rscl, img_r_rscl, img_i_rscl


def generate_rgb(src_path, dst_path, number=20000):
    # scales, offset, Q, alpha, masklevel, saturation,
    # itype = (0.75,1.05,1.5),
    # 0.01, 2.0,0.7, -1.0, 'color', 'rms' # gri
    scales, offset, Q, alpha = (0.75, 1.05, 1.5), 0.01, 2.0, 0.7
    masklevel = -1.0
    saturation, itype = 'white', 'sum'
    start = 100000
    bands = ['Band2', 'Band1', 'Band3', 'Band4']
    band_letter = ['I', 'R', 'G', 'U']
    for i in range(number):
        image_src = []
        for j in range(3):
            image = f'{bands[j]}/imageSDSS_{band_letter[j]}-{start + i}.fits'
            image = os.path.join(src_path, image)
            image_src.append(image)
        img_g_rscl, img_r_rscl, img_i_rscl = img_preproc(*image_src)

        dst_image = f'ground_based_{start + i}.png'
        dst_image = os.path.join(dst_path, dst_image)

        object_gri = channel_RGB(RED=img_i_rscl, GREEN=img_r_rscl, BLUE=img_g_rscl)
        object_gri.apply_scale(scales=scales)
        object_gri.lupton_stretch(Q=Q, alpha=alpha, itype=itype)
        object_gri.pjm_mask(masklevel=masklevel)
        object_gri.pjm_offset(offset=offset)
        object_gri.lupton_saturate(saturation=saturation)
        object_gri.pack_up()
        object_gri.imgRGB.save(dst_image)
        print(f"{dst_image} saved.")


if __name__ == "__main__":
    generate_rgb(sys.argv[1], sys.argv[2])
