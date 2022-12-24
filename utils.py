import rasterio
import numpy as np
import cv2


def ensure_opened(ds):
    "Ensure that `ds` is an opened Rasterio dataset and not a str/pathlike object."
    return ds if type(ds) == rasterio.io.DatasetReader else rasterio.open(str(ds), "r")

def read_crop(ds, crop, bands=None, pad=False):
    """
    Read rasterio `crop` for the given `bands`..
    Args:
        ds: Rasterio dataset.
        crop: Tuple or list containing the area to be cropped (px, py, w, h).
        bands: List of `bands` to read from the dataset.
    Returns:
        A numpy array containing the read image `crop` (bands * h * w).
    """
    ds = ensure_opened(ds)
    if pad: raise ValueError('padding not implemented yet.')
    if bands is None:
        bands = [i for i in range(1, ds.count+1)]

    #assert len(bands) <= ds.count, "`bands` cannot contain more bands than the number of bands in the dataset."
    #assert max(bands) <= ds.count, "The maximum value in `bands` should be smaller or equal to the band count."
    window = None
    if crop is not None:
        assert len(crop) == 4, "`crop` should be a tuple or list of shape (px, py, w, h)."
        px, py, w, h = crop
        w = ds.width - px if (px + w) > ds.width else w
        h = ds.height - py if (py + h) > ds.height else h
        assert (px + w) <= ds.width, "The crop (px + w) is larger than the dataset width."
        assert (py + h) <= ds.height, "The crop (py + h) is larger than the dataset height."
        window = rasterio.windows.Window(px, py, w, h)
    meta = ds.meta
    meta.update(count=len(bands))
    if crop is not None:
        meta.update({
        'height': window.height,# make the aoi more smooth so data is easier correctly downloaded
        'width': window.width,
        'transform': rasterio.windows.transform(window, ds.transform)})
    return ds.read(bands, window=window), meta




def plot_rgb(img, clip_percentile=(2, 98), clip_values=None, bands=[3, 2, 1], figsize=(20, 20), nodata=None, figtitle=None, crop=None, ax=None):
    """
    Plot clipped (and optionally cropped) RGB image.
    Args:
        img: Path to image, rasterio dataset or numpy array of shape (bands, height, width).
        clip_percentile: (min percentile, max percentile) to use for clippping.
        clip_values: (min value, max value) to use for clipping (if set clip_percentile is ignored).
        bands: Bands to use as RGB values (starting at 1).
        figsize: Size of the matplotlib figure.
        figtitle: Title to use for the figure (if None and img is a path we will use the image filename).
        crop: Window to use to crop the image (px, py, w, h).
        ax: If not None, use this Matplotlib axis for plotting.
    Returns:
        A matplotlib figure.
    """
    meta = None
    if isinstance(img, str):
        assert os.path.exists(img), "{} does not exist!".format(img)
        figtitle = os.path.basename(img) if figtitle is None else figtitle
        img = rasterio.open(img)
        img, meta = read_crop(img, crop, bands)
    elif isinstance(img, rasterio.io.DatasetReader):
        img, meta = read_crop(img, crop, bands)
    elif isinstance(img, np.ndarray):
        assert len(img.shape) <= 3, "Array should have no more than 3 dimensions."
        if len(img.shape) == 2:
            img = img[np.newaxis, :, :]
        elif img.shape[0] > 3:
            img = img[np.array(bands) - 1, :, :]
        if crop is not None:
            img = img[:, py:py+h, px:px+w]
    else:
        raise ValueError("img should be str, rasterio dataset or numpy array. (got {})".format(type(img)))
    img = img.astype(float)
    nodata = nodata if nodata is not None else (meta['nodata'] if meta is not None else None)
    if nodata is not None:
        img[img == nodata] = np.nan
    if clip_values is not None:
        assert len(clip_values) == 2, "Clip values should have the shape (min value, max value)"
        assert clip_values[0] < clip_values[1], "clip_values[0] should be smaller than clip_values[1]"
    elif clip_percentile is not None:
        assert len(clip_percentile) == 2, "Clip_percentile should have the shape (min percentile, max percentile)"
        assert clip_percentile[0] < clip_percentile[1], "clip_percentile[0] should be smaller than clip_percentile[1]"
        clip_values = None if clip_percentile == (0, 100) else [np.nanpercentile(img, clip_percentile[i]) for i in range(2)]
    if clip_values is not None:
        img[~np.isnan(img)] = np.clip(img[~np.isnan(img)], *clip_values)
    clip_values = (np.nanmin(img), np.nanmax(img)) if clip_values is None else clip_values
    img[~np.isnan(img)] = (img[~np.isnan(img)] - clip_values[0])/(clip_values[1] - clip_values[0])
    if img.shape[0] <= 3:
        img = np.transpose(img, (1, 2, 0))
    alpha = np.all(~np.isnan(img), axis=2)[:,:,np.newaxis].astype(float)
    img = np.concatenate((img, alpha), axis=2)

    if not ax:
        figure, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_title(figtitle) if figtitle is not None else None
        ax.imshow(img)
        plt.close()
        return figure
    else:
        ax.imshow(img)