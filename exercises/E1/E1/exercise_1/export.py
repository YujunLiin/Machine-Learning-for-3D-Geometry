"""Export to disk"""


def export_mesh_to_obj(path, vertices, faces):
    """
    exports mesh as OBJ
    :param path: output path for the OBJ file
    :param vertices: Nx3 vertices
    :param faces: Mx3 faces
    :return: None
    """

    # write vertices starting with "v "
    # write faces starting with "f "

    # ###############
    # TODO: Implement
    #raise NotImplementedError
    # ###############
    with open(path,'w')as file:
        for vertex in range(vertices.shape[0]):
            print(f"v {float(vertices[vertex][0])} {float(vertices[vertex][1])} {float(vertices[vertex][2])}",file=file)
        for face in range(faces.shape[0]):
            print(f"f {int(faces[face][0])} {int(faces[face][1])} {int(faces[face][2])}",file=file)


def export_pointcloud_to_obj(path, pointcloud):
    """
    export pointcloud as OBJ
    :param path: output path for the OBJ file
    :param pointcloud: Nx3 points
    :return: None
    """

    # ###############
    # TODO: Implement
    #raise NotImplementedError
    # ###############
    with open(path,'w')as file:
        for point in range(pointcloud.shape[0]):
            print(f"v {pointcloud[point,0]} {pointcloud[point,1]} {pointcloud[point,2]}",file=file)

