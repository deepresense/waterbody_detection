import rasterio

with rasterio.open('tile_37.tif') as src:
    has_transform = src.transform != rasterio.Affine.identity()
    has_crs = src.crs is not None
    has_gcps = bool(src.gcps[0])

    if has_transform or has_gcps:
        print("Image contains georeference data.")

        # Get first 5 row pixel coordinates and convert to spatial coordinates
        for row in range(5):
            for col in range(src.width):
                x, y = src.transform * (col, row)
                print(f"Row {row}, Col {col} => X: {x}, Y: {y}")
            break  # only print the first row (remove this `break` if you want 5 full rows)
    else:
        print("Image does NOT contain georeference data.")
