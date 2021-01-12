import torch
import torch.nn as nn
from threading import Thread
from models.layers.mesh_union import MeshUnion
import numpy as np
from heapq import heappop, heapify
from models.layers.mesh_prepare import get_edge_faces
import math


class MeshPool(nn.Module):

    def __init__(self, target, multi_thread=False):
        super(MeshPool, self).__init__()
        self.__out_target = target
        self.__multi_thread = multi_thread
        self.__fe = None
        self.__updated_fe = None
        self.__meshes = None
        self.__merge_edges = [-1, -1]

    def __call__(self, fe, meshes):
        return self.forward(fe, meshes)

    def forward(self, fe, meshes):
        self.__updated_fe = [[] for _ in range(len(meshes))]
        pool_threads = []
        self.__fe = fe

        self.__meshes = meshes
        # iterate over batch
        for mesh_index in range(len(meshes)):
            if self.__multi_thread:
                pool_threads.append(Thread(target=self.__pool_main, args=(mesh_index,)))
                pool_threads[-1].start()
            else:
                self.__pool_main(mesh_index)
        if self.__multi_thread:
            for mesh_index in range(len(meshes)):
                pool_threads[mesh_index].join()
        out_features = torch.cat(self.__updated_fe).view(len(meshes), -1, self.__out_target)
        return out_features

    def __pool_main(self, mesh_index):

        mesh = self.__meshes[mesh_index]
        QEM, New_vertice = self.build_QEM(mesh, mesh.edges_count)
        queue = self.__build_queue1(QEM, mesh.edges_count)
        last_count = mesh.edges_count + 1
        mask = np.ones(mesh.edges_count, dtype=np.bool)
        edge_groups = MeshUnion(mesh.edges_count, self.__fe.device)
        while mesh.edges_count > self.__out_target:
            value, edge_id = heappop(queue)
            edge_id = int(edge_id)
            newvertice = New_vertice[edge_id]
            if newvertice == []:
                continue
            # print('len',edge_id, len(newvertice),file=data)
            if mask[edge_id]:
                self.__pool_edge(mesh, edge_id, mask, edge_groups, newvertice)

        # print('after', mesh.edges_count)
        mesh.clean(mask, edge_groups)
        fe = edge_groups.rebuild_features(self.__fe[mesh_index], mask, self.__out_target)
        #fe=extract_features1(mask,mesh)
        #fe1 = edge_groups.rebuild_features(fe, mask, self.__out_target)
        self.__updated_fe[mesh_index] = fe




    def __pool_edge(self, mesh, edge_id, mask, edge_groups, new_vertex):
        if self.has_boundaries(mesh, edge_id):
            return False
        elif self.__clean_side(mesh, edge_id, mask, edge_groups, 0) \
                and self.__clean_side(mesh, edge_id, mask, edge_groups, 2) \
                and self.__is_one_ring_valid(mesh, edge_id):
            self.__merge_edges[0] = self.__pool_side(mesh, edge_id, mask, edge_groups, 0)
            self.__merge_edges[1] = self.__pool_side(mesh, edge_id, mask, edge_groups, 2)
            mesh.merge_vertices(edge_id, new_vertex)
            mask[edge_id] = False
            MeshPool.__remove_group(mesh, edge_groups, edge_id)
            mesh.edges_count -= 1
            return True
        else:
            return False



    # 首先，gemm是不是跟着faces变了
    # 第二个问题，是不是在折叠的时候，gemm改变了，是不是顺着我们的想法改变的
    # 第三个问题，为什么在原来的程序中，没有这种问题，需要打印出来求证
    # 第四个，解决方法，如果以上方式都不奏效，那么可以利用build_gemm重构edge,face等数组
    @staticmethod
    def __pool_face(mesh):


        np.set_printoptions(threshold=10000000000000000)

        faces = set()
        faces2=[]
        gemm = np.array(mesh.gemm_edges)


        for edge_index in range(len(gemm)):
            gem = gemm[edge_index]
            for i in range(2):
                otheredge = [gem[i * 2], gem[i * 2 + 1]]
                face = set()
                face.add(mesh.edges[otheredge[0]][0])
                face.add(mesh.edges[otheredge[0]][1])
                face.add(mesh.edges[otheredge[1]][0])
                face.add(mesh.edges[otheredge[1]][1])

                face=list(face)
                face.sort()
                face_normals = np.cross(mesh.vs[face[1]] - mesh.vs[face[0]],
                                        mesh.vs[face[2]] - mesh.vs[face[1]])
                face_areas = np.sqrt((face_normals ** 2).sum())
                if face_areas==0.0:
                    print(face)
                face2 = []
                face=tuple(face)
                face2.append(face)
                face2=set(face2)
                #print('face',face2)
                faces=faces|face2
                #print('faces',faces)
                # if face in faces:
                #     continue
                # else:
                #     faces.append(face)

        #edge_count, edge_faces, edges_dict = get_edge_faces(faces)
        #edge_count, edges, edges_dict=MeshPool.get_edge(faces)
        #    print(abcbdbdb)

        #mesh.edges = edges
        faces = list(faces)
        return faces#, edge_count, edges_dict

    @staticmethod
    def __get_edge(mesh):
        edge_count = 0
        edge2keys = dict()
        for i in range(len(mesh.edges)):
            cur_edge=tuple(sorted((mesh.edges[i][0],mesh.edges[i][1])))
           # if cur_edge not in edge2keys:
            edge2keys[cur_edge] = edge_count
            edge_count += 1
        return edge_count, edge2keys

    def __clean_side(self, mesh, edge_id, mask, edge_groups, side):
        if mesh.edges_count <= self.__out_target:
            return False
        invalid_edges = MeshPool.__get_invalids(mesh, edge_id, edge_groups, side)
        while len(invalid_edges) != 0 and mesh.edges_count > self.__out_target:
            self.__remove_triplete(mesh, mask, edge_groups, invalid_edges)
            if mesh.edges_count <= self.__out_target:
                return False
            if self.has_boundaries(mesh, edge_id):
                return False
            invalid_edges = self.__get_invalids(mesh, edge_id, edge_groups, side)
        return True

    @staticmethod
    def has_boundaries(mesh, edge_id):
        for edge in mesh.gemm_edges[edge_id]:
            if edge == -1 or -1 in mesh.gemm_edges[edge]:
                return True
        return False

    @staticmethod
    def __is_one_ring_valid(mesh, edge_id):
        v_a = set(mesh.edges[mesh.ve[mesh.edges[edge_id, 0]]].reshape(-1))
        v_b = set(mesh.edges[mesh.ve[mesh.edges[edge_id, 1]]].reshape(-1))
        shared = v_a & v_b - set(mesh.edges[edge_id])
        return len(shared) == 2

    def __pool_side(self, mesh, edge_id, mask, edge_groups, side):
        info = MeshPool.__get_face_info(mesh, edge_id, side)
        key_a, key_b, side_a, side_b, _, other_side_b, _, other_keys_b = info
        self.__redirect_edges(mesh, key_a, side_a - side_a % 2, other_keys_b[0], mesh.sides[key_b, other_side_b])
        self.__redirect_edges(mesh, key_a, side_a - side_a % 2 + 1, other_keys_b[1],
                              mesh.sides[key_b, other_side_b + 1])
        MeshPool.__union_groups(mesh, edge_groups, key_b, key_a)
        MeshPool.__union_groups(mesh, edge_groups, edge_id, key_a)
        mask[key_b] = False
        MeshPool.__remove_group(mesh, edge_groups, key_b)
        mesh.remove_edge(key_b)
        mesh.edges_count -= 1
        return key_a

    @staticmethod
    def __get_invalids(mesh, edge_id, edge_groups, side):
        info = MeshPool.__get_face_info(mesh, edge_id, side)
        key_a, key_b, side_a, side_b, other_side_a, other_side_b, other_keys_a, other_keys_b = info
        shared_items = MeshPool.__get_shared_items(other_keys_a, other_keys_b)
        if len(shared_items) == 0:
            return []
        else:
            assert (len(shared_items) == 2)
            middle_edge = other_keys_a[shared_items[0]]
            update_key_a = other_keys_a[1 - shared_items[0]]
            update_key_b = other_keys_b[1 - shared_items[1]]
            update_side_a = mesh.sides[key_a, other_side_a + 1 - shared_items[0]]
            update_side_b = mesh.sides[key_b, other_side_b + 1 - shared_items[1]]
            MeshPool.__redirect_edges(mesh, edge_id, side, update_key_a, update_side_a)
            MeshPool.__redirect_edges(mesh, edge_id, side + 1, update_key_b, update_side_b)
            MeshPool.__redirect_edges(mesh, update_key_a, MeshPool.__get_other_side(update_side_a), update_key_b,
                                      MeshPool.__get_other_side(update_side_b))
            MeshPool.__union_groups(mesh, edge_groups, key_a, edge_id)
            MeshPool.__union_groups(mesh, edge_groups, key_b, edge_id)
            MeshPool.__union_groups(mesh, edge_groups, key_a, update_key_a)
            MeshPool.__union_groups(mesh, edge_groups, middle_edge, update_key_a)
            MeshPool.__union_groups(mesh, edge_groups, key_b, update_key_b)
            MeshPool.__union_groups(mesh, edge_groups, middle_edge, update_key_b)
            return [key_a, key_b, middle_edge]

    @staticmethod
    def __redirect_edges(mesh, edge_a_key, side_a, edge_b_key, side_b):
        mesh.gemm_edges[edge_a_key, side_a] = edge_b_key
        mesh.gemm_edges[edge_b_key, side_b] = edge_a_key
        mesh.sides[edge_a_key, side_a] = side_b
        mesh.sides[edge_b_key, side_b] = side_a

    @staticmethod
    def __get_shared_items(list_a, list_b):
        shared_items = []
        for i in range(len(list_a)):
            for j in range(len(list_b)):
                if list_a[i] == list_b[j]:
                    shared_items.extend([i, j])
        return shared_items

    @staticmethod
    def __get_other_side(side):
        return side + 1 - 2 * (side % 2)

    @staticmethod
    def __get_face_info(mesh, edge_id, side):
        key_a = mesh.gemm_edges[edge_id, side]
        key_b = mesh.gemm_edges[edge_id, side + 1]
        side_a = mesh.sides[edge_id, side]
        side_b = mesh.sides[edge_id, side + 1]
        other_side_a = (side_a - (side_a % 2) + 2) % 4
        other_side_b = (side_b - (side_b % 2) + 2) % 4
        other_keys_a = [mesh.gemm_edges[key_a, other_side_a], mesh.gemm_edges[key_a, other_side_a + 1]]
        other_keys_b = [mesh.gemm_edges[key_b, other_side_b], mesh.gemm_edges[key_b, other_side_b + 1]]
        return key_a, key_b, side_a, side_b, other_side_a, other_side_b, other_keys_a, other_keys_b

    @staticmethod
    def __remove_triplete(mesh, mask, edge_groups, invalid_edges):
        vertex = set(mesh.edges[invalid_edges[0]])
        for edge_key in invalid_edges:
            vertex &= set(mesh.edges[edge_key])
            mask[edge_key] = False
            MeshPool.__remove_group(mesh, edge_groups, edge_key)
        mesh.edges_count -= 3
        vertex = list(vertex)
        assert (len(vertex) == 1)
        mesh.remove_vertex(vertex[0])

    def __build_queue(self, features, edges_count):
        # delete edges with smallest norm
        squared_magnitude = torch.sum(features * features, 0)
        if squared_magnitude.shape[-1] != 1:
            squared_magnitude = squared_magnitude.unsqueeze(-1)
        edge_ids = torch.arange(edges_count, device=squared_magnitude.device, dtype=torch.float32).unsqueeze(-1)
        heap = torch.cat((squared_magnitude, edge_ids), dim=-1).tolist()
        heapify(heap)

        return heap

    def __build_queue1(self, QEM, edges_count):
        QEMM = torch.tensor(QEM, device=self.__fe.device, dtype=torch.float32).unsqueeze(-1)
        edge_ids = torch.arange(edges_count, device=self.__fe.device, dtype=torch.float32).unsqueeze(-1)
        heap = torch.cat((QEMM, edge_ids), dim=-1).tolist()
        heapify(heap)

        return heap

    def build_QEM(self, mesh, edges_count):  # 与update_mesh一致

        QEM = [0 for i in range(edges_count)]

        Newvertice = [[0,0,0] for i in range(edges_count)]

        Q = [[0 for i in range(10)] for j in range(len(mesh.vs))]
        faces = MeshPool.__pool_face(mesh)
        edge_count, edges_dict=  MeshPool.__get_edge(mesh)

        np.set_printoptions(threshold=10000000000000000)
        for i in range(len(faces)):
            p = []
            face = faces[i]
            #print('face',face)
            # print(face)
            for j in range(3):

                p.append(mesh.vs[face[j]])
            new1 = MeshPool.cross(p[1] - p[0], p[2] - p[0])
            if (new1[0] == 0):
                continue
            new = MeshPool.normarlize(new1)
            if (math.isnan(new[0])):
                continue

            q1 = MeshPool.getm(new[0], new[1], new[2], -(new[0] * p[0][0] + new[1] * p[0][1] + new[2] * p[0][2]))

            for j in range(3):
                for k in range(10):
                    if math.isnan(q1[k]):
                        print('q1chuwentile')
                    Q[face[j]][k] = Q[face[j]][k] + q1[k]

        for f in faces:
            p_result = [0, 0, 0]
            for i in range(3):
                err, new_vertex = MeshPool.calculate_error(mesh, Q, f[i], f[(i + 1) % 3], p_result)

                edge = tuple(sorted((f[i], f[(i + 1) % 3])))
                edges=mesh.edges.tolist()

                if edge in mesh.edges:  # 问题出在这
                    edge_id = edges_dict[edge]
                    QEM[edge_id] = err
                    if Newvertice[edge_id][0]==0.0:
                        Newvertice[edge_id][0] = new_vertex[0]
                        Newvertice[edge_id][1] = new_vertex[1]
                        Newvertice[edge_id][2] = new_vertex[2]
                    else:
                        continue

        return QEM, Newvertice

    @staticmethod
    def __vertex_error(q, x, y, z):
        error = q[0] * x * x + 2 * q[1] * x * y + 2 * q[2] * x * z + 2 * q[3] * x + q[4] * y * y \
                + 2 * q[5] * y * z + 2 * q[6] * y + q[7] * z * z + 2 * q[8] * z + q[9]
        return error

    @staticmethod
    def calculate_error(mesh, Q, id_v1, id_v2, p_result):

        q = [0 for i in range(10)]
        for i in range(10):
            q[i] = Q[id_v1][i] + Q[id_v2][i]

        d = MeshPool.det(q, 0, 1, 2, 1, 4, 5, 2, 5, 7)

        if d != 0:
            p_result[0] = -1 / d * (MeshPool.det(q, 1, 2, 3, 4, 5, 6, 5, 7, 8))
            p_result[1] = 1 / d * (MeshPool.det(q, 0, 2, 3, 1, 5, 6, 2, 7, 8))
            p_result[2] = -1 / d * (MeshPool.det(q, 0, 1, 3, 1, 4, 6, 2, 5, 8))

            error = MeshPool.__vertex_error(q, p_result[0], p_result[1], p_result[2])

        else:
            p1 = mesh.vs[id_v1]
            p2 = mesh.vs[id_v2]
            p3 = [0, 0, 0]
            p3[0] = (p1[0] + p2[0]) / 2
            p3[1] = (p1[1] + p2[1]) / 2
            p3[2] = (p1[2] + p2[2]) / 2
            error1 = MeshPool.__vertex_error(q, p1[0], p1[1], p1[2])
            error2 = MeshPool.__vertex_error(q, p2[0], p2[1], p2[2])
            error3 = MeshPool.__vertex_error(q, p3[0], p3[1], p3[2])
            error = min(error1, min(error2, error3))
            if error == error1:
                p_result[0] = p1[0]
                p_result[1] = p1[1]
                p_result[2] = p1[2]
            if error == error2:
                p_result[0] = p2[0]
                p_result[1] = p2[1]
                p_result[2] = p2[2]
            if error == error3:
                p_result[0] = p3[0]
                p_result[1] = p3[1]
                p_result[2] = p3[2]

        return error, p_result

    @staticmethod
    def min(a, b):
        if (a < b):
            return a
        else:
            return b

    @staticmethod
    def cross(a, b):
        x = a[1] * b[2] - a[2] * b[1]
        y = a[2] * b[0] - a[0] * b[2]
        z = a[0] * b[1] - a[1] * b[0]
        new = []
        new.append(x)
        new.append(y)
        new.append(z)
        return new

    @staticmethod
    def normarlize(a):
        square = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) ** 0.5
        a[0] = a[0] / square
        a[1] = a[1] / square
        a[2] = a[2] / square
        return a

    @staticmethod
    def getm(*args):
        m = [0 for i in range(10)]

        if len(args) == 10:
            for i in range(10):
                m[i] = args[i]

        elif len(args) == 4:
            m[0] = (args[0] * args[0])
            m[1] = (args[0] * args[1])
            m[2] = (args[0] * args[2])
            m[3] = (args[0] * args[3])
            m[4] = (args[1] * args[1])
            m[5] = (args[1] * args[2])
            m[6] = (args[1] * args[3])
            m[7] = (args[2] * args[2])
            m[8] = (args[2] * args[3])
            m[9] = (args[3] * args[3])

        if math.isnan(m[0]):
            print('yuanshuju', args)
        return m

    @staticmethod
    def det(*args):

        m = args[0]
        a11 = args[1]
        a12 = args[2]
        a13 = args[3]
        a21 = args[4]
        a22 = args[5]
        a23 = args[6]
        a31 = args[7]
        a32 = args[8]
        a33 = args[9]

        det = m[a11] * m[a22] * m[a33] + m[a13] * m[a21] * m[a32] \
              + m[a12] * m[a23] * m[a31] - m[a13] * m[a22] * m[a31] \
              - m[a11] * m[a23] * m[a32] - m[a12] * m[a21] * m[a33]
        return det

    @staticmethod
    def operationadd(m, n):
        return MeshPool.getm(m[0] + n[0], m[1] + n[1], m[2] + n[2],
                             m[3] + n[3], m[4] + n[4], m[5] + n[5], m[6] + n[6], m[7] + n[7], m[8] + n[8],
                             m[9] + n[9])

    @staticmethod
    def __union_groups(mesh, edge_groups, source, target):
        edge_groups.union(source, target)
        mesh.union_groups(source, target)

    @staticmethod
    def __remove_group(mesh, edge_groups, index):
        edge_groups.remove_group(index)
        mesh.remove_group(index)

