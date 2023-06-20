""" Procrustes Aligment for point clouds """
import numpy as np
from pathlib import Path


# def procrustes_align(pc_x, pc_y):
#     """
#     calculate the rigid transform to go from point cloud pc_x to point cloud pc_y, assuming points are corresponding
#     :param pc_x: Nx3 input point cloud
#     :param pc_y: Nx3 target point cloud, corresponding to pc_x locations
#     :return: rotation (3, 3) and translation (3,) needed to go from pc_x to pc_y
#     """
#     R = np.zeros((3, 3), dtype=np.float32)
#     t = np.zeros((3,), dtype=np.float32)

#     # TODO: Your implementation starts here ###############
#     # 1. get centered pc_x and centered pc_y
#     # 2. create X and Y both of shape 3XN by reshaping centered pc_x, centered pc_y
#     # 3. estimate rotation
#     # 4. estimate translation
#     # R and t should now contain the rotation (shape 3x3) and translation (shape 3,)
#     centroid_x=np.mean(pc_x,axis=0)
#     centered_x=pc_x-centroid_x
#     centroid_y=np.mean(pc_y,axis=0)
#     centered_y=pc_y-centroid_y

#     X=centered_x.T
#     Y=centered_y.T
#     cross_covariance=X@Y.T
#     U,S,V_T=np.linalg.svd(cross_covariance)
#     R=V_T.T@U.T
#     t=centroid_y-np.dot(R,centroid_x)
#     # TODO: Your implementation ends here ###############

#     t_broadcast = np.broadcast_to(t[:, np.newaxis], (3, pc_x.shape[0]))
#     print('Procrustes Aligment Loss: ', np.abs((np.matmul(R, pc_x.T) + t_broadcast) - pc_y.T).mean())

#     return R, t

def procrustes_align(pc_x, pc_y):
    """
    calculate the rigid transform to go from point cloud pc_x to point cloud pc_y, assuming points are corresponding
    :param pc_x: Nx3 input point cloud
    :param pc_y: Nx3 target point cloud, corresponding to pc_x locations
    :return: rotation (3, 3) and translation (3,) needed to go from pc_x to pc_y
    """
    R = np.zeros((3, 3), dtype=np.float32)
    t = np.zeros((3,), dtype=np.float32)

    # TODO: Your implementation starts here ###############
    N,K  = pc_x.shape
    # 1. get centered pc_x and centered pc_y
    sx = np.mean(pc_x, axis = 0)
    sy = np.mean(pc_y,axis = 0)
    # 2. create X and Y both of shape 3XN by reshaping centered pc_x, centered pc_y
    X,Y = pc_x-sx, pc_y-sy
    # 3. estimate rotation
    X,Y = np.array(X),np.array(Y)
    matrix = np.dot(X.T,Y)
    U, D, V = np.linalg.svd(matrix)
    det_U = np.linalg.det(U)
    det_V = np.linalg.det(V)
    S = np.eye(K)
    if det_U*det_V != 1:
        S[K-1,K-1] = -1

    R = np.dot(V.T@S, U.T)
    # 4. estimate translation
    t = sy - np.dot(R, sx)
    # R and t should now contain the rotation (shape 3x3) and translation (shape 3,)
    



    # TODO: Your implementation ends here ###############

    t_broadcast = np.broadcast_to(t[:, np.newaxis], (3, pc_x.shape[0]))
    print('Procrustes Aligment Loss: ', np.abs((np.matmul(R, pc_x.T) + t_broadcast) - pc_y.T).mean())

    return R, t


def load_correspondences():
    """
    loads correspondences between meshes from disk
    """

    load_obj_as_np = lambda path: np.array(list(map(lambda x: list(map(float, x.split(' ')[1:4])), path.read_text().splitlines())))
    path_x = (Path(__file__).parent / "resources" / "points_input.obj").absolute()
    path_y = (Path(__file__).parent / "resources" / "points_target.obj").absolute()
    return load_obj_as_np(path_x), load_obj_as_np(path_y)
