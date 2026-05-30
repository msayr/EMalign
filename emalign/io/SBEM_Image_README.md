Extract relative tile positions formatted as (y, x) from imagelist metadata file. For instance, 
    in a slice with a single grid (grid 0) that has 2 tiles (1 row 2 columns), the left tile position is (0, 0) and the 
    right tile position is (0, 1). If a second grid with a single tile were to be added below grid 0, 
    the first two tiles maintain the same position as above and the new grid tile would have position (1, 0). 

    Because the relative position of the grids can be arbitrary, the most reliable way to determined 
    tile position is by using the exact tile coordinates.
    
    1. Extract filename of the image with the lowest z slice ('sxxxxx') for each tile using get_tilesets(). For instance, 
    if a directory contains only three images named: 

    <project>_g0001_t0002_s00003.tif
    <project>_g0001_t0002_s00004.tif
    <project>_g0001_t0002_s00005.tif

    then the function extracts <project>_g0001_t0002_s00003.tif. 
    
    2. Using the extracted filenames as keys, locate entry in meta/logs/imagelist_<datetime>.txt that matches the filename. 
    There will be many imagelist files to check, but the correct file will likely be the first chronologically. Here is an 
    example entry in imagelist_<datetime>.txt: 

    tiles\g0000\t0000\<project>_g0000_t0000_s00000.tif;-608632;-619022;0;0

    Where:    
    . - 608632 = X coordinate 
    . - 619022 = Y coordinate
    . 0 = Z coordinate: 0 nm (slice 0)
    . 0 = Slice counter: slice 0

    The two values to extract are the X and Y coordinate which indicate the top-left corner of the image tile. 
    These coordinates are then compared to all other image tiles to roughly determine approximate relative tile positions. 

    Here is an example for a project that has three image tiles. The first two are part of grid 0
    and the second is grid 1.  

    tiles\g0000\t0000\<project>_g0000_t0000_s00000.tif;-608632;-619022;0;0
    tiles\g0000\t0001\<project>_g0000_t0001_s00000.tif;-421332;-619022;0;0
    tiles\g0001\t0000\<project>_g0001_t0000_s00000.tif;-683129;-493211;0;0

    Resulting grid shape:

    "tiles\\g0000\\t0000\\<project>_g0000_t0000_s00000.tif": (0, 1),
    "tiles\\g0000\\t0001\\<project>_g0000_t0001_s00000.tif": (0, 2),
    "tiles\\g0001\\t0000\\<project>_g0001_t0000_s00000.tif": (1, 0)


