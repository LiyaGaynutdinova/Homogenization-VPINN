import numpy as np
import torch 

def compute_bound(T1, dq1, tri, data, A, H, L):
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
        U1 = T1[elem]
        d1 = D_grad @ U1
        Q1 = A_loc @ (d1.view(1,2) + H(node_coords)).view(3,2,1)
        Q1_mean = Q1.mean(dim=0)
        A_h_elem_1 = (area * Q1_mean).squeeze().detach().cpu().numpy()
        A_h_linear[:,0] = A_h_linear[:,0] + A_h_elem_1

    return A_h / L**2, A_h_linear / L**2


def compute_avg(dq1, data, A, H, L):
    A_loc = A(data)
    H_loc = H(data)
    Q = A_loc @ (dq1 + H_loc).view(-1,2,1)
    Q_mean = Q.mean(dim=0)
    return Q_mean.squeeze().detach().cpu().numpy()