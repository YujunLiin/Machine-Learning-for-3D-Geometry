"""Triangle Meshes to Point Clouds"""
import numpy as np


def sample_point_cloud(vertices, faces, n_points):
    """
    Sample n_points uniformly from the mesh represented by vertices and faces
    :param vertices: Nx3 numpy array of mesh vertices
    :param faces: Mx3 numpy array of mesh faces
    :param n_points: number of points to be sampled
    :return: sampled points, a numpy array of shape (n_points, 3)
    """

    # ###############
    # TODO: Implement
    #raise NotImplementedError
    # ###############

    def compute_area(vertex1,vertex2,vertex3):
        temp=vertex1[0]*(vertex2[1]*vertex3[2]-vertex3[1]*vertex2[2])+\
             vertex2[0]*(vertex3[1]*vertex1[2]-vertex1[1]*vertex3[2])+\
             vertex3[0]*(vertex1[1]*vertex2[2]-vertex2[1]*vertex1[2])
        return 0.5*np.abs(temp)
    
    def compute_point(vertex1,vertex2,vertex3):
        r1,r2=np.random.rand(2)
        u=1-np.sqrt(r1)
        v=np.sqrt(r1)*(1-r2)
        w=np.sqrt(r1)*r2
        return u*vertex1+v*vertex2+w*vertex3
    
    area_sum=0.0
    total_points=np.zeros((faces.shape[0],3))
    area=np.zeros(faces.shape[0])
    for face in range(faces.shape[0]):
        area[face]=compute_area(vertices[faces[face][0]],vertices[faces[face][1]],vertices[faces[face][2]])
        total_points[face]=compute_point(vertices[faces[face][0]],vertices[faces[face][1]],vertices[faces[face][2]])
        area_sum+=area[face]
    
    probability=area/area_sum
    points_index=np.random.choice(np.arange(faces.shape[0]),size=n_points,p=probability)
    return total_points[points_index]



