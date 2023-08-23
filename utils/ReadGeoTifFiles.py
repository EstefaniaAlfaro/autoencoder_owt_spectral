import os
from PIL import Image
from tifffile import tifffile

PATH_DATA = '../data'


class ReadGeoTifFiles:
    def __init__(self, image_tif_file):
        self.image_tif_file = image_tif_file

    def read_geo_tif_file(self):
        full_path = os.path.join(PATH_DATA, self.image_tif_file)
        image_geo_tif = tifffile.imread(full_path)
        return image_geo_tif
