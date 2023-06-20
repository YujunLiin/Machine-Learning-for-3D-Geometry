"""SDF to Occupancy Grid"""
import numpy as np


def occupancy_grid(sdf_function, resolution):
    """
    Create an occupancy grid at the specified resolution given the implicit representation.
    :param sdf_function: A function that takes in a point (x, y, z) and returns the sdf at the given point.
    Points may be provides as vectors, i.e. x, y, z can be scalars or 1D numpy arrays, such that (x[0], y[0], z[0])
    is the first point, (x[1], y[1], z[1]) is the second point, and so on
    :param resolution: Resolution of the occupancy grid
    :return: An occupancy grid of specified resolution (i.e. an array of dim (resolution, resolution, resolution) with value 0 outside the shape and 1 inside.
    """

    # ###############
    # TODO: Implement
    #raise NotImplementedError
    # ###############
    sdf=np.zeros((resolution,resolution,resolution))
    unit_cube_length=1
    unit_cube_shift=np.array([-0.5,-0.5,-0.5])
    voxel_length=unit_cube_length/resolution
    x=[]
    y=[]
    z=[]

    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                x.append(i*voxel_length+unit_cube_shift[0])
                y.append(j*voxel_length+unit_cube_shift[1])
                z.append(k*voxel_length+unit_cube_shift[2])

    x=np.array(x)
    y=np.array(y)
    z=np.array(z)

    sdf=sdf_function(x,y,z)<=0
    occpancy_grid=sdf.reshape((resolution,resolution,resolution))
    return occpancy_grid

