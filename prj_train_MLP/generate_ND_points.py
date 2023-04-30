import numpy as np
import os
import logging
import cv2
import matplotlib.pyplot as plt
import math

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s [%(levelname)s] %(message)s')

def f0(values):
    y = 0
    """for dim_ndx in range(len(values)):
        exponent = -1. + -2. * dim_ndx + 0.00000 * dim_ndx**2
        coef = -600. - 1000. * dim_ndx
        #logging.debug(f"exponent = {exponent}; coef = {coef}; values[dim_ndx] = {values[dim_ndx]}")
        y += (values[dim_ndx]/abs(values[dim_ndx])) * coef * math.pow(abs(values[dim_ndx]), exponent)
    """
    for dim_ndx in range(len(values)):
        coef = -3. + dim_ndx
        y += 1.5 * math.sin(coef * values[dim_ndx])
    return y

def f1(values):
    y = 0
    """for dim_ndx in range(len(values)):
        exponent = -1. + dim_ndx - 0.01 * dim_ndx**2
        coef = 100. - dim_ndx
        y += (values[dim_ndx]/abs(values[dim_ndx])) * coef * math.pow(abs(values[dim_ndx]), exponent)
    """
    for dim_ndx in range(len(values)):
        coef = -3. + dim_ndx
        y += 0.35 * math.cos(coef * values[dim_ndx])
    return y

def f2(values):
    y = 0
    """for dim_ndx in range(len(values)):
        exponent = 4. + 2 * dim_ndx - 0.01 * dim_ndx**2
        coef = 100. + dim_ndx - 0.01 * dim_ndx**2
        y += (values[dim_ndx]/abs(values[dim_ndx])) * coef * math.pow(abs(values[dim_ndx]), exponent)
    """
    for dim_ndx in range(len(values)):
        coef = 1. - dim_ndx
        y += 0.35 * math.cos(coef * values[dim_ndx])
    return y

def main():
    logging.info("generate_ND_points.main()")
    output_directory = "./output_generate_ND_points"
    if not os.path.exists((output_directory)):
        os.makedirs(output_directory)

    number_of_points = 6000
    number_of_dimensions = 6



    # draw random numbers
    class_0_number = 0
    class_1_number = 1
    class_2_number = 2
    with open(os.path.join(output_directory, "dataset.csv"), 'w') as output_file:
        #output_file.write("v0,v1,class\n")
        for dim_ndx in range(number_of_dimensions):
            output_file.write(f"v{dim_ndx},")
        output_file.write("class\n")
        for pt_ndx in range(number_of_points):
            v = np.random.randn(number_of_dimensions)
            f0_v = f0(v)
            f1_v = f1(v)
            f2_v = f2(v)
            class_ndx = None
            if f0_v > f1_v and f0_v > f2_v:
                class_ndx = 0
                class_0_number += 1
            elif f1_v > f0_v and f1_v > f2_v:
                class_ndx = 1
                class_1_number += 1
            else:
                class_ndx = 2
                class_2_number += 1

            # Move the point randomly to blur the boundary

            for dim_ndx in range(len(v)):
                output_file.write(f"{v[dim_ndx]},")
            output_file.write(f"{class_ndx}\n")
    logging.info(f"class_0_number = {class_0_number}; class_1_number = {class_1_number}; class_2_number = {class_2_number}")

if __name__ == '__main__':
    main()