import os
import sys
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


if __name__ == "__main__":
    pass
