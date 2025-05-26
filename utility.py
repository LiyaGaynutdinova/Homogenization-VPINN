import numpy as np
import torch 

def compute_bound(T1, dq1, tri, data, A, H, L, type='primal'):
    # Compute the A_h
    A_h = np.zeros((2,1))
    A_h_linear = np.zeros((2,1))
    n_elem = len(tri.triangles)
    
    for i in range(n_elem):
        elem = tri.triangles[i]
        node_coords = data[elem]
        vector = node_coords[1:]-node_coords[:-1]
        vectors = torch.hstack((vector, torch.tensor([[0], [0]], device=T1.device)))
        area = 0.5 * abs(torch.cross(vectors[0], vectors[1])[-1])
        A_loc = A(node_coords)

        # use the gradients given by the NN
        d1_nn = dq1[elem]
        Q1 = A_loc @ (d1_nn + H(node_coords)).view(3,2,1)
        Q1_mean = Q1.mean(dim=0)
        A_h_elem_1 = (area * Q1_mean).squeeze().detach().cpu().numpy()
        A_h[:,0] += A_h_elem_1

        # use the gradients given by the linear approximation
        M = torch.column_stack((torch.ones(3, device=T1.device), node_coords))
        M_inv = torch.linalg.inv(M)
        D_grad = M_inv[1:]
        if type=='dual':
            D_curl = torch.zeros_like(D_grad)
            D_curl[0] = -D_grad[1]
            D_curl[1] = D_grad[0]
            D_grad = D_curl
        U1 = T1[elem]
        d1 = D_grad @ U1
        Q1 = A_loc @ (d1.view(1,2) + H(node_coords)).view(3,2,1)
        Q1_mean = Q1.mean(dim=0)
        A_h_elem_1 = (area * Q1_mean).squeeze().detach().cpu().numpy()
        A_h_linear[:,0] = A_h_linear[:,0] + A_h_elem_1

    if type=='primal':
            return A_h / L**2, A_h_linear / L**2
    else:
        B_l = np.zeros((2,2))
        B_l[:, 0] = A_h.flatten()
        B_l[:, 1] = A_h.flatten()[::-1]
        A_h_l = np.linalg.inv(B_l / L**2)
        B_lin = np.zeros((2,2))
        B_lin[:, 0] = A_h_linear.flatten()
        B_lin[:, 1] = A_h_linear.flatten()[::-1]
        A_h_l_linear = np.linalg.inv(B_lin / L**2) 
        return A_h_l, A_h_l_linear

    


def compute_avg(dq1, data, A, H):
    A_loc = A(data)
    H_loc = H(data)
    Q = A_loc @ (dq1 + H_loc).view(-1,2,1)
    Q_mean = Q.mean(dim=0)
    return Q_mean.squeeze().detach().cpu().numpy()


