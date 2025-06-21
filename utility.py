import numpy as np
import torch 

def compute_bound_primal(T1, T2, tri, data, A, L):
    U = torch.column_stack([T1, T2])
    A_h = np.zeros((2,2))
    n_elem = len(tri.triangles)
    H = torch.eye(2, device=T1.device)
    
    for i in range(n_elem):
        elem = tri.triangles[i]
        node_coords_x1 = data[elem, 0]
        node_coords_x2 = data[elem, 1]
        vector_x1 = node_coords_x1[1:]-node_coords_x1[:-1]
        vector_x2 = node_coords_x2[1:]-node_coords_x2[:-1]
        vectors = torch.column_stack((vector_x1, vector_x2, torch.tensor([0, 0], device=T1.device)))
        area = 0.5 * abs(torch.cross(vectors[0], vectors[1])[-1])
        center = torch.column_stack([node_coords_x1.mean(), node_coords_x2.mean()])
        A_loc = A(center).squeeze()

        # use the gradients given by the linear approximation
        M = torch.column_stack((torch.ones(3, device=T1.device), node_coords_x1, node_coords_x2))
        M_inv = torch.linalg.inv(M)
        D_grad = M_inv[1:]
        g_elem = D_grad @ U[elem] + H
        A_h_elem = area * g_elem.T @ A_loc @ g_elem
        A_h = A_h + A_h_elem.detach().cpu().numpy()

    return A_h / L**2
    

def compute_bound_dual(T1, T2, tri, data, A, L):
    W = torch.column_stack([T1, T2])
    B_h = np.zeros((2,2))
    n_elem = len(tri.triangles)
    H = torch.eye(2, device=T1.device)
    
    for i in range(n_elem):
        elem = tri.triangles[i]
        node_coords_x1 = data[elem, 0]
        node_coords_x2 = data[elem, 1]
        vector_x1 = node_coords_x1[1:]-node_coords_x1[:-1]
        vector_x2 = node_coords_x2[1:]-node_coords_x2[:-1]
        vectors = torch.column_stack((vector_x1, vector_x2, torch.tensor([0, 0], device=T1.device)))
        area = 0.5 * abs(torch.cross(vectors[0], vectors[1])[-1])
        center = torch.column_stack([node_coords_x1.mean(), node_coords_x2.mean()])
        A_loc = A(center).squeeze()

        # use the gradients given by the linear approximation
        M = torch.column_stack((torch.ones(3, device=T1.device), node_coords_x1, node_coords_x2))
        M_inv = torch.linalg.inv(M)
        D_grad = M_inv[1:]
        D_curl = torch.zeros_like(D_grad)
        D_curl[0] = -D_grad[1]
        D_curl[1] = D_grad[0]
        curl = D_curl @ W[elem] + H
        B_h_elem = area * curl.T @ A_loc @ curl
        B_h = B_h + B_h_elem.detach().cpu().numpy()
    
    A_l = np.linalg.inv(B_h / L**2)

    return A_l

    
def compute_avg(dq1, data, A, H):
    A_loc = A(data)
    H_loc = H(data)
    Q = A_loc @ (dq1 + H_loc).view(-1,2,1)
    Q_mean = Q.mean(dim=0)
    return Q_mean.squeeze().detach().cpu().numpy()


def compute_bound_p(T1, tri, data, A, L):
    A_h = 0
    n_elem = len(tri.triangles)
    H = torch.tensor([[1.], [0.]], device=T1.device)
    
    for i in range(n_elem):
        elem = tri.triangles[i]
        node_coords_x1 = data[elem, 0]
        node_coords_x2 = data[elem, 1]
        vector_x1 = node_coords_x1[1:]-node_coords_x1[:-1]
        vector_x2 = node_coords_x2[1:]-node_coords_x2[:-1]
        vectors = torch.column_stack((vector_x1, vector_x2, torch.tensor([0, 0], device=T1.device)))
        area = 0.5 * abs(torch.cross(vectors[0], vectors[1])[-1])
        center = torch.column_stack([node_coords_x1.mean(), node_coords_x2.mean()])
        A_loc = A(center).squeeze()

        # use the gradients given by the linear approximation
        M = torch.column_stack((torch.ones(3, device=T1.device), node_coords_x1, node_coords_x2))
        M_inv = torch.linalg.inv(M)
        D_grad = M_inv[1:]
        g_elem = D_grad @ T1[elem] + H
        A_h_elem = area * g_elem.T @ A_loc @ g_elem
        A_h = A_h + A_h_elem.detach().cpu().squeeze().numpy()

    return A_h / L**2


def compute_bound_d(T1, tri, areas, data, A, L):
    B_h = 0
    n_elem = len(tri.triangles)
    H = torch.tensor([[1.], [0.]], device=T1.device)
    
    for i in range(n_elem):
        elem = tri.triangles[i]
        node_coords_x1 = data[elem, 0]
        node_coords_x2 = data[elem, 1]
        area = areas[i]
        center = torch.column_stack([node_coords_x1.mean(), node_coords_x2.mean()])
        A_loc = A(center).squeeze()

        # use the gradients given by the linear approximation
        M = torch.column_stack((torch.ones(3, device=T1.device), node_coords_x1, node_coords_x2))
        M_inv = torch.linalg.inv(M)
        D_grad = M_inv[1:]
        D_curl = torch.zeros_like(D_grad)
        D_curl[0] = -D_grad[1]
        D_curl[1] = D_grad[0]
        curl = D_curl @ T1[elem] + H
        B_h_elem = area * curl.T @ A_loc @ curl
        B_h = B_h + B_h_elem.detach().cpu().squeeze().numpy()
    
    A_l = 1 / (B_h / L**2)

    return A_l