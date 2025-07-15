import rasterio

with rasterio.open('multicolor label.tif') as src:
    has_transform = src.transform != rasterio.Affine.identity()
    has_crs = src.crs is not None
    has_gcps = bool(src.gcps[0])

    if has_transform or has_gcps:
        print("Image contains georeference data.")
    else:
        print("Image does NOT contain georeference data.")
