# image to svg converter
# loic landrieu 2020

import numpy as np
import sys, os
import matplotlib.image as mpimg
import argparse
import ast

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_path, "grid-graph/python/bin"))
sys.path.append(os.path.join(file_path, "parallel-cut-pursuit/python/wrappers"))
sys.path.append(os.path.join(file_path, "multilabel-potrace/python/bin"))

from grid_graph import grid_to_graph
from cp_d0_dist import cp_d0_dist
from multilabel_potrace_svg import multilabel_potrace_svg


def tostr(f):
    return "%5.1f" % f

def tochar(f):
    return int(255 * f)

def char2col(c):
    switcher = {
        "r": [255,0,0],
        "g": [0,255,0],
        "b": [0,0,255],
        "k": [0, 0, 0],
        "w": [255, 255, 255]
    }
    return np.array(switcher.get(c)).astype('uint8')

def main():
    parser = argparse.ArgumentParser(description='IMG TO VECTOR')
    # path and filenames
    parser.add_argument('-f', '--file', default='lola.jpeg', required=False,
                        help='Input file name.')
    parser.add_argument('-o', '--outfile', default='',
                        help='Path of SVG outputfile. Default: <input>.svg')
    #cosmetic
    parser.add_argument('-lw', '--line_width', default=1, type=int,
                        help='Width of contours in pixels. Set to 0 for no ' \
                             'contour. Default: 1')
    parser.add_argument('-lc', '--line_color', default='k',
                        help='Color of contour. Supported: r,g,b,k,w, or a ' \
                            'char triplet. Defaut: k.')
    # optimization parameters
    parser.add_argument('-a', '--apply', default='',
                        help='Function to apply before partition: sqrt, log,' \
                             ' none (default).')
    parser.add_argument('-r', '--reg', default=1.0, type=float,
                        help='Regularization strength: the higher the fewer' \
                             ' components. Default: 1.0.')
    parser.add_argument('-m', '--min_comp_size', default=10, type=int,
                        help='Minimum size of components (in pixels). ' \
                             'Default: 10')
    parser.add_argument('-s', '--smooth', default=1.0, type=float,
                        help='Smoothing term. [0, 4/3]; 0 = polygonal, ' \
                             '> 0 cubic Bezier curves. Default: 1.0')
    parser.add_argument('-lt', '--line_tolerance', default=1.0, type=float,
                        help='How far are lines allowed to deviate from the ' \
                             'borders (in pixels). Default: 1.0.')
    parser.add_argument('-ct', '--curve_tolerance', default=0.2, type=float,
                        help='Max difference area ratio diff between ' \
                             'original and simplified polygons. Default: 0.2')

    args = parser.parse_args()

    if len(args.outfile) > 3 and args.outfile[-4:] != '.svg':
        args.outfile = args.outfile + '.svg'

    # input raster
    filename, file_extension = os.path.splitext(args.file)
    if file_extension in '.png.jpg.jpeg':
        img = mpimg.imread(args.file).astype('f4')
        if file_extension == '.png':
            img = img[:, :, :3]
        if img.max() > 1:
            img = img / 255.0
    elif file_extension == '.npy':
        img = np.load(args.file).astype('f4')[:, :, 0]
    elif file_extension in '.tif.tiff':
        from PIL import Image
        img = np.array(Image.open(args.file)).astype('f4')
        img[img != img] = 0.0
        nodata = True
        img = img / img.max()
    else:
        raise NotImplementedError('unknown file extension %s' % file_extension)

    if 'log' in args.apply:
        print("Log mapping")
        img = np.log(np.maximum(img, 0) + 1e-4)
    elif 'sqrt' in args.apply:
        print("Square root mapping")
        img = np.sqrt(np.maximum(img, 0))

    args.lin = img.shape[0]
    args.col = img.shape[1]
    args.n_chan = img.shape[-1] if len(img.shape) > 2 else 1
    args.n_ver = args.lin * args.col
    print('Loaded image of size %d by %d with %d channels' %
            (args.lin, args.col, args.n_chan))

    # compute grid graph
    shape = np.array([args.lin, args.col], dtype='uint32')
    first_edge, adj_vertices, connectivities = grid_to_graph(shape, 2,
        compute_connectivities=True)
    # edge weights
    edge_weights = np.ones(connectivities.shape, dtype=img.dtype)
    edge_weights[connectivities == 2] = 1 / np.sqrt(2)
    del connectivities

    # cut pursuit
    reg_strength = args.reg * np.var(img)

    img = np.asfortranarray(img.reshape((args.n_ver, args.n_chan)).T)

    comp, rX = cp_d0_dist(1,
            img, first_edge, adj_vertices,
            edge_weights=reg_strength * edge_weights, cp_it_max=10,
            min_comp_weight=args.min_comp_size, cp_dif_tol=1e-2,
            max_num_threads=0, balance_parallel_split=True)

    print('Partition done.')

    if 'log' in args.apply:
        rX = np.exp(rX)
    if 'sqrt' in args.apply:
        rX = rX ** 2

    # format output
    output_path = args.outfile if len(args.outfile) > 0 \
                               else filename + '.svg'
    if len(args.line_color) == 0:
        line_color = None
    elif len(args.line_color) == 1:
        line_color = char2col(args.line_color)
    else:
        try:
            line_color = np.array(ast.literal_eval(args.line_color)) \
                .astype('uint8')
        except SyntaxError:
            print("line_color should be either empty, r,g,b,k,w or a char" \
                  " triplet.")

    multilabel_potrace_svg(np.resize(comp, (args.lin, args.col)), output_path,
        straight_line_tol=args.line_tolerance, smoothing=args.smooth,
        curve_fusion_tol=args.curve_tolerance,
        comp_colors=(255 * rX).astype('uint8'), line_color=line_color,
        line_width=args.line_width)

    print('Vectorization done.')

if __name__ == "__main__":
    main()
