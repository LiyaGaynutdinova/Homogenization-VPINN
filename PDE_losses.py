import torch
from matplotlib.tri import Triangulation

def PDE_loss(x, net, a_function, H_function):
    # Ensure x has requires_grad; clone to avoid modifying original tensor
    x = x.clone().detach().requires_grad_(True)
    
    # Forward pass to get y (batch_size, 1)
    y = net(x)
    
    # Compute ∇y: (batch_size, 2)
    grad_y = torch.autograd.grad(
        outputs=y, inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True
    )[0]
    
    # H + ∇y (broadcast H to batch_size, 2)
    H_plus_grad = H_function(x) + grad_y  # H shape (2,) -> (batch_size, 2)
    
    # Compute K(x) as (batch_size, 2, 2)
    K = a_function(x)
    
    # Compute q = K @ (H + ∇y) using batched matrix multiplication
    q = torch.bmm(K, H_plus_grad.unsqueeze(-1)).squeeze(-1)  # (batch_size, 2)
    
    # Compute divergence of q: sum of dq_i/dx_i
    divergence = torch.zeros_like(y)  # (batch_size, 1)
    for i in range(2):
        # Get q_i component (batch_size,)
        q_i = q[:, i]
        
        # Compute gradient of q_i w.r.t. x
        dq_i = torch.autograd.grad(
            outputs=q_i, inputs=x,
            grad_outputs=torch.ones_like(q_i),
            create_graph=True, retain_graph=True
        )[0]  # (batch_size, 2)
        
        # Accumulate the i-th component's derivative
        divergence += dq_i[:, i].unsqueeze(-1)
    
    return divergence**2, q


def PDE_loss_dual(x, net, a_function, H):
    # Ensure x has requires_grad; clone to avoid modifying original tensor
    x = x.clone().detach().requires_grad_(True)
    
    # Forward pass to get y (batch_size, 1)
    u = net(x)
    
    # Compute ∇y: (batch_size, 2)
    du = torch.autograd.grad(
        outputs=u, inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]

    curl_u = torch.zeros_like(du)
    curl_u[:,0] = -du[:,1]
    curl_u[:,1] = du[:,0]
   
    # H + ∇y (broadcast H to batch_size, 2)
    H_plus_grad = H(x) + curl_u  # H shape (2,) -> (batch_size, 2)
   
    # Compute K(x) as (batch_size, 2, 2)
    R = a_function(x)
    
    # Compute q = K @ (H + ∇y) using batched matrix multiplication
    q = torch.bmm(R, H_plus_grad.unsqueeze(-1)).squeeze(-1)  # (batch_size, 2)
   
    # Compute gradient of q_1 w.r.t. x
    dq1 = torch.autograd.grad(
        outputs=q[:,0], inputs=x,
        grad_outputs=torch.ones_like(q[:,0]),
        create_graph=True, retain_graph=True
    )[0]  # (batch_size, 2)

    # Compute gradient of q_2 w.r.t. x
    dq2 = torch.autograd.grad(
        outputs=q[:,1], inputs=x,
        grad_outputs=torch.ones_like(q[:,1]),
        create_graph=True, retain_graph=True
    )[0]  # (batch_size, 2)

    curl = dq2[:,0] - dq1[:,1]
    
    return curl**2, q


def get_areas(x):
    areas = []
    triang = Triangulation(x[:,0], x[:,1])
    n_elem = len(triang.triangles)
    for i in range(n_elem):
        elem = triang.triangles[i]
        node_coords_x1 = x[elem,0]
        node_coords_x2 = x[elem,1]
        vector_x1 = node_coords_x1[1:]-node_coords_x1[:-1]
        vector_x2 = node_coords_x2[1:]-node_coords_x2[:-1]
        vectors = torch.column_stack((vector_x1, vector_x2, torch.zeros((2,1))))
        area = (1/2) * abs(torch.cross(vectors[0], vectors[1])[-1])
        areas.append(area.detach())
    return torch.tensor(areas), triang


def compute_bound(areas, tri, Q_loc, L):
    # Compute the bound
    elem = tri.triangles
    Q_tri = Q_loc[elem]
    prod_mean = Q_tri.mean(dim=1)
    tri_int = areas.unsqueeze(1) * prod_mean
    A_h = tri_int.sum(dim=0)
    
    return A_h / L**2


def weak_loss_avg(x, net, test_functions, a_function, H_function, G_inv, Lx):
    # Ensure x has requires_grad; clone to avoid modifying original tensor
    x = x.clone().detach().requires_grad_(True)
    
    # Forward pass to get y (batch_size, 1)
    y = net(x)
    
    # Compute ∇y: (batch_size, 2)
    grad_y = torch.autograd.grad(
        outputs=y, inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True
    )[0]

    # H + ∇y (broadcast H to batch_size, 2)
    H_plus_grad = H_function(x) + grad_y  # H shape (2,) -> (batch_size, 2)
    
    # Compute K(x) as (batch_size, 2, 2)
    K = a_function(x)
    
    # Compute q = K @ (H + ∇y) using batched matrix multiplication
    q = torch.bmm(K, H_plus_grad.unsqueeze(-1)).squeeze(-1)  # (batch_size, 2)
    
    n_test = len(test_functions)
    residual = torch.zeros(n_test, device=x.device)

    for i in range(n_test):
        grad_test = test_functions[i]
        test_prod = torch.bmm(q.view(-1,1,2), grad_test.view(-1,2,1))
        residual[i] = Lx**2 * test_prod.mean()

    L = residual.view(1, n_test) @ G_inv @ residual.view(n_test, 1)

    return L



def compute_int(areas, tri, fun_loc, test):
    elem = tri.triangles
    f_tri = fun_loc[elem]
    test_tri = test[elem]
    prod = torch.matmul(test_tri.view(-1,3,1,2), f_tri.view(-1,3,2,1)).squeeze()
    prod_mean = prod.mean(dim=1)
    tri_int = areas * prod_mean
    intgr = tri_int.sum()
    return intgr


def weak_loss_primal(x, net, areas, tri, test_functions, a_function, H_function, G_inv):
    # Ensure x has requires_grad; clone to avoid modifying original tensor
    x = x.clone().detach().requires_grad_(True)
    
    # Forward pass to get y (batch_size, 1)
    y = net(x)
    
    # Compute ∇y: (batch_size, 2)
    grad_y = torch.autograd.grad(
        outputs=y, inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True
    )[0]

    # H + ∇y (broadcast H to batch_size, 2)
    H_plus_grad = H_function(x) + grad_y  # H shape (2,) -> (batch_size, 2)
    
    # Compute K(x) as (batch_size, 2, 2)
    K = a_function(x)
    
    # Compute q = K @ (H + ∇y) using batched matrix multiplication
    q = torch.bmm(K, H_plus_grad.unsqueeze(-1)).squeeze(-1)  # (batch_size, 2)

    n_test = len(test_functions)
    residual = torch.zeros(n_test, device=x.device)

    for i in range(n_test):
        grad_test = test_functions[i]
        residual[i] = compute_int(areas, tri, q, grad_test)
   
    L = residual.view(1, n_test) @ G_inv @ residual.view(n_test, 1)

    return L, q


def weak_loss_dual(x, net, areas, tri, test_functions, a_function, H_function, G_inv):
    # Ensure x has requires_grad; clone to avoid modifying original tensor
    x = x.clone().detach().requires_grad_(True)
    
    # Forward pass to get y (batch_size, 1)
    y = net(x)
    
    # Compute ∇y: (batch_size, 2)
    grad_y = torch.autograd.grad(
        outputs=y, inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True
    )[0]
    curl_u = torch.zeros_like(grad_y)
    curl_u[:,0] = -grad_y[:,1]
    curl_u[:,1] = grad_y[:,0]

    # H + ∇y (broadcast H to batch_size, 2)
    H_plus_grad = H_function(x) + curl_u  # H shape (2,) -> (batch_size, 2)
    
    # Compute K(x) as (batch_size, 2, 2)
    R = a_function(x)
    
    # Compute q = R @ (H + ∇y) using batched matrix multiplication
    q = torch.bmm(R, H_plus_grad.unsqueeze(-1)).squeeze(-1)  # (batch_size, 2)

    n_test = len(test_functions)
    residual = torch.zeros(n_test, device=x.device)

    for i in range(n_test):
        grad_test = test_functions[i]
        residual[i] = compute_int(areas, tri, q, grad_test)
   
    L = residual.view(1, n_test) @ G_inv @ residual.view(n_test, 1)

    return L, q


def divergence(x, net, a_function, H_function):
    # Ensure x has requires_grad; clone to avoid modifying original tensor
    x = x.clone().detach().requires_grad_(True)
    
    # Forward pass to get y (batch_size, 1)
    y = net(x)
    
    # Compute ∇y: (batch_size, 2)
    grad_y = torch.autograd.grad(
        outputs=y, inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True
    )[0]
    
    # H + ∇y (broadcast H to batch_size, 2)
    H_plus_grad = H_function(x) + grad_y  # H shape (2,) -> (batch_size, 2)
    
    # Compute K(x) as (batch_size, 2, 2)
    K = a_function(x)
    
    # Compute q = K @ (H + ∇y) using batched matrix multiplication
    q = torch.bmm(K, H_plus_grad.unsqueeze(-1)).squeeze(-1)  # (batch_size, 2)
    
    # Compute divergence of q: sum of dq_i/dx_i
    divergence = torch.zeros_like(y)  # (batch_size, 1)
    for i in range(2):
        # Get q_i component (batch_size,)
        q_i = q[:, i]
        
        # Compute gradient of q_i w.r.t. x
        dq_i = torch.autograd.grad(
            outputs=q_i, inputs=x,
            grad_outputs=torch.ones_like(q_i),
            create_graph=True, retain_graph=True
        )[0]  # (batch_size, 2)
        
        # Accumulate the i-th component's derivative
        divergence += dq_i[:, i].unsqueeze(-1)
    
    return divergence


def strong_loss(x, net, test_functions, a_function, H_function, G_inv, Lx):
    # Ensure x has requires_grad; clone to avoid modifying original tensor
    x = x.clone().detach().requires_grad_(True)

    div_u = divergence(x, net, a_function, H_function).squeeze()
    
    n_test = len(test_functions)
    residual = torch.zeros(n_test, device=x.device)

    for i in range(n_test):
        test = test_functions[i]
        test_prod = div_u * test
        residual[i] = Lx**2 * test_prod.mean()
   
    L = residual.view(1, n_test) @ G_inv @ residual.view(n_test, 1)

    return L


def PDE_loss_primal_dual(x, net, a_function, H_function):
    # Ensure x has requires_grad; clone to avoid modifying original tensor
    x = x.clone().detach().requires_grad_(True)
    
    # Forward pass to get y (batch_size, 1)
    y = net(x)
    
    # Compute ∇y: (batch_size, 2)
    grad_y_primal = torch.autograd.grad(
        outputs=y[:,0], inputs=x,
        grad_outputs=torch.ones_like(y[:,0]),
        create_graph=True, retain_graph=True
    )[0]

    grad_y_dual = torch.autograd.grad(
        outputs=y[:,1], inputs=x,
        grad_outputs=torch.ones_like(y[:,1]),
        create_graph=True, retain_graph=True
    )[0]
    curl_u = torch.zeros_like(grad_y_dual)
    curl_u[:,0] = -grad_y_dual[:,1]
    curl_u[:,1] = grad_y_dual[:,0]
    
    # H + ∇y (broadcast H to batch_size, 2)
    H_plus_grad = H_function(x) + grad_y_primal  # H shape (2,) -> (batch_size, 2)

    # H + ∇y (broadcast H to batch_size, 2)
    H_plus_curl = H_function(x) + curl_u  # H shape (2,) -> (batch_size, 2)
    
    # Compute K(x) as (batch_size, 2, 2)
    K = a_function(x)
    K_inv = torch.linalg.inv(K)
    
    # Compute q = K @ (H + ∇y) using batched matrix multiplication
    q_p = torch.bmm(K, H_plus_grad.unsqueeze(-1)).squeeze(-1)  # (batch_size, 2)
    q_d = torch.bmm(K_inv, H_plus_curl.unsqueeze(-1)).squeeze(-1)  # (batch_size, 2)
    
    # Compute divergence/curl of q: sum of dq_i/dx_i
    dq1_p = torch.autograd.grad(
        outputs=q_p[:,0], inputs=x,
        grad_outputs=torch.ones_like(q_p[:,0]),
        create_graph=True, retain_graph=True
    )[0]  # (batch_size, 2)

    dq2_p = torch.autograd.grad(
        outputs=q_p[:,1], inputs=x,
        grad_outputs=torch.ones_like(q_p[:,1]),
        create_graph=True, retain_graph=True
    )[0]  # (batch_size, 2)

    divergence = dq1_p[:,0] + dq2_p[:,1]
    
    # Compute gradient of q_1 w.r.t. x
    dq1_d = torch.autograd.grad(
        outputs=q_d[:,0], inputs=x,
        grad_outputs=torch.ones_like(q_d[:,0]),
        create_graph=True, retain_graph=True
    )[0]  # (batch_size, 2)

    # Compute gradient of q_2 w.r.t. x
    dq2_d = torch.autograd.grad(
        outputs=q_d[:,1], inputs=x,
        grad_outputs=torch.ones_like(q_d[:,1]),
        create_graph=True, retain_graph=True
    )[0]  # (batch_size, 2)

    curl = dq2_d[:,0] - dq1_d[:,1]
    
    return divergence**2, curl**2, q_p, q_d


def weak_loss_primal_dual(x, net, areas, tri, test_functions_primal, test_functions_dual, a_function, H_function, G_inv):
    # Ensure x has requires_grad; clone to avoid modifying original tensor
    x = x.clone().detach().requires_grad_(True)
    
    # Forward pass to get y (batch_size, 1)
    y = net(x)
    
    # Compute ∇y: (batch_size, 2)
    grad_y_primal = torch.autograd.grad(
        outputs=y[:,0], inputs=x,
        grad_outputs=torch.ones_like(y[:,0]),
        create_graph=True, retain_graph=True
    )[0]

    grad_y_dual = torch.autograd.grad(
        outputs=y[:,1], inputs=x,
        grad_outputs=torch.ones_like(y[:,1]),
        create_graph=True, retain_graph=True
    )[0]
    curl_u = torch.zeros_like(grad_y_dual)
    curl_u[:,0] = -grad_y_dual[:,1]
    curl_u[:,1] = grad_y_dual[:,0]
    
    # H + ∇y (broadcast H to batch_size, 2)
    H_plus_grad = H_function(x) + grad_y_primal  # H shape (2,) -> (batch_size, 2)

    # H + ∇y (broadcast H to batch_size, 2)
    H_plus_curl = H_function(x) + curl_u  # H shape (2,) -> (batch_size, 2)
    
    # Compute K(x) as (batch_size, 2, 2)
    K = a_function(x)
    K_inv = torch.linalg.inv(K)
    
    # Compute q = K @ (H + ∇y) using batched matrix multiplication
    q_p = torch.bmm(K, H_plus_grad.unsqueeze(-1)).squeeze(-1)  # (batch_size, 2)
    q_d = torch.bmm(K_inv, H_plus_curl.unsqueeze(-1)).squeeze(-1)  # (batch_size, 2)
    
    n_test = len(test_functions_primal)
    residual_p = torch.zeros(n_test, device=x.device)

    for i in range(n_test):
        grad_test = test_functions_primal[i]
        residual_p[i] = compute_int(areas, tri, q_p, grad_test)

    n_test = len(test_functions_dual)
    residual_d = torch.zeros(n_test, device=x.device)

    for i in range(n_test):
        grad_test = test_functions_dual[i]
        residual_d[i] = compute_int(areas, tri, q_d, grad_test)
   
    L_p = residual_p.view(1, -1) @ G_inv @ residual_p.view(-1, 1)
    L_d = residual_d.view(1, -1) @ G_inv @ residual_d.view(-1, 1)
    
    return L_p, L_d, q_p, q_d


def compute_G_grad(x, areas, tri, test_functions):
    n_test = len(test_functions)

    # Ensure x has requires_grad; clone to avoid modifying original tensor
    x = x.clone().detach().requires_grad_(True)
    
    # Forward pass to get y (batch_size, 1)
    ys = [test_functions[i](x) for i in range(n_test)] 

    grads = []
    for i in range(n_test):
        # Compute ∇y: (batch_size, 2)
        grad_y = torch.autograd.grad(
            outputs=ys[i], inputs=x,
            grad_outputs=torch.ones_like(ys[i]),
            create_graph=True, retain_graph=True
        )[0]
        grads.append(grad_y)
    
    G = torch.zeros((n_test, n_test), device=x.device)
    for i in range(n_test):
        for j in range(i, n_test):
            intgr = compute_int(areas, tri, grads[i], grads[j])
            G[i,j] = intgr
  
    return G


def compute_G(x, areas, tri, test_functions):
    n_test = len(test_functions)

    # Ensure x has requires_grad; clone to avoid modifying original tensor
    x = x.clone().detach().requires_grad_(True)
    
    # Forward pass to get y (batch_size, 1)
    ys = torch.zeros((n_test, x.shape[0], 2, 1), device=x.device)
    for i in range(n_test):
        ys[i,:,0] = test_functions[i](x)

    G = torch.zeros((n_test, n_test), device=x.device)
    for i in range(n_test):
        for j in range(i, n_test):
            intgr = compute_int(areas, tri, ys[i], ys[j])
            G[i,j] = intgr
  
    return G