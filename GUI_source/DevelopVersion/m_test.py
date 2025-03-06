def bilinear_interpolate_cupy(map, x, y):
    """
    Perform bilinear interpolation for CuPy arrays.
    map: The 2D CuPy array on which interpolation is performed.
    x: x-coordinates (float) for interpolation.
    y: y-coordinates (float) for interpolation.
    """
    # Get integer coordinates surrounding the point
    x0 = cp.clip(cp.floor(x).astype(cp.int32), 0, map.shape[1] - 2)
    y0 = cp.clip(cp.floor(y).astype(cp.int32), 0, map.shape[0] - 2)
    
    x1, y1 = x0 + 1, y0 + 1

    # Use cp.take_along_axis for advanced indexing
    Ia, Ib, Ic, Id = map[y0, x0], map[y0, x1], map[y1, x0], map[y1, x1]

    # Interpolation weights
    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)

    # Calculate the interpolated value
    return wa * Ia + wb * Ib + wc * Ic + wd * Id