import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import torch

def stiff_tensor(E, p):
    """Linear stiffness tensor for an isotropic material"""
    D = E / ((1 + p) * (1 - 2 * p)) * np.array([[1 - p, p, 0], [p, 1 - p, 0], [0, 0, (1 - 2 * p) / 2]]); 
    return D


def stiff_tensor(E, p):
    """Linear stiffness tensor for an isotropic material"""
    D = E / ((1 + p) * (1 - 2 * p)) * np.array([[1 - p, p, 0], [p, 1 - p, 0], [0, 0, (1 - 2 * p) / 2]]); 
    return D


def create_domain(E, p, t_x, t_y, res=8):
    """Create local stiffness matrix for a module, E1 - Young's modulus, [Pa]; p - Poisson's constant, [-];
    t_x - thickness in horizontal direction, 1<=t_x<=res/2; 
    t_y - thickness in vertical direction; 1<=t_y<=res/2;
    res - resolution of the module"""

    x, y = np.meshgrid(np.linspace(0, 1, res+1), np.linspace(0, 1, res+1))
    x = x.flatten()
    y = y.flatten()

    # Create triangular elements (with empty elements)
    elem_grid = []
    for i in range(res*(res+1)):
        if i % (res+1) != res:
            elem_grid.append([i, i+1, i+res+2])
            elem_grid.append([i, i+res+1, i+res+2])
    
    # Construct module's stiffness matrix
    K = np.zeros((2*(res+1)**2, 2*(res+1)**2))
    for i in range(len(elem_grid)):
        elem = elem_grid[i]
        loc = [2 * elem[0], 2 * elem[0] + 1, 2 * elem[1], 2 * elem[1] + 1, 2 * elem[2], 2 * elem[2] + 1]
        n_x = x[elem]
        n_y = y[elem]
        center = np.array([np.mean(n_x), np.mean(n_y)])
        D = stiff_tensor(E(center), p)
        # Check if the element is not empty
        vectors = np.vstack((n_x[1:] - n_x[:-1], n_y[1:] - n_y[:-1]))
        area = 0.5 * abs(np.cross(vectors[:,0], vectors[:,1]))
        B_e = 1 / (2 * area) * np.array([[n_y[1] - n_y[2], 0, n_y[2] - n_y[0], 0, n_y[0] - n_y[1], 0],
                                         [0, n_x[2] - n_x[1], 0, n_x[0] - n_x[2], 0, n_x[1] - n_x[0]],
                                         [n_x[2] - n_x[1], n_y[1] - n_y[2], n_x[0] - n_x[2], 
                                          n_y[2] - n_y[0], n_x[1] - n_x[0], n_y[0] - n_y[1]]])
        K_elem = area * np.matmul(B_e.T, np.matmul(D, B_e))
        K[np.ix_(loc, loc)] += K_elem
    
    return K


def create_module_domain_from_png(E1, E2, p, name):
    """Create local stiffness matrix for a module, E1 - Young's modulus, [Pa]; E2 - ersatz Young's modulus, [Pa]; p - Poisson's constant, [-];
    
    res - resolution of the module"""
    image = plt.imread(name)[:,:,0].round().T
    res = image.shape[0]
    x, y = np.meshgrid(np.linspace(0, 1, res+1), np.linspace(0, 1, res+1))
    x = x.flatten()
    y = y.flatten()

    # Create triangular elements
    elem_grid = []
    for i in range(res*(res+1)):
        if i % (res+1) != res:
            elem_grid.append([i, i+1, i+res+2])
            elem_grid.append([i, i+res+1, i+res+2])

    D = stiff_tensor(E1, p)
    D_0 = stiff_tensor(E2, p)
    
    # Construct module's stiffness matrix
    K = np.zeros((2*(res+1)**2, 2*(res+1)**2))
    for i in range(len(elem_grid)):
        elem = elem_grid[i]
        loc = [2 * elem[0], 2 * elem[0] + 1, 2 * elem[1], 2 * elem[1] + 1, 2 * elem[2], 2 * elem[2] + 1]
        n_x = x[elem]
        n_y = y[elem]
        center = np.array([np.mean(n_x), np.mean(n_y)])
        # Check if the element is not empty
        vectors = np.vstack((n_x[1:] - n_x[:-1], n_y[1:] - n_y[:-1]))
        area = 0.5 * abs(np.cross(vectors[:,0], vectors[:,1]))
        B_e = 1 / (2 * area) * np.array([[n_y[1] - n_y[2], 0, n_y[2] - n_y[0], 0, n_y[0] - n_y[1], 0],
                                         [0, n_x[2] - n_x[1], 0, n_x[0] - n_x[2], 0, n_x[1] - n_x[0]],
                                         [n_x[2] - n_x[1], n_y[1] - n_y[2], n_x[0] - n_x[2], 
                                          n_y[2] - n_y[0], n_x[1] - n_x[0], n_y[0] - n_y[1]]])
        color = image[int(center[0]*res), int(center[1]*res)]
        if color == 0:
            K_elem = area * np.matmul(B_e.T, np.matmul(D, B_e))
        else:
            K_elem = area * np.matmul(B_e.T, np.matmul(D_0, B_e))
        K[np.ix_(loc, loc)] += K_elem
    
    return K


def assemble_global_stiffness(K, layout, coarse_grid, n_nodes_glob, res):
    K_glob = np.zeros((2*n_nodes_glob, 2*n_nodes_glob))
    glob_numbering = np.reshape(np.arange(n_nodes_glob), (coarse_grid[0]*res+1, coarse_grid[1]*res+1))

    for i in range(len(layout)):
        glob_view = glob_numbering[(i//coarse_grid[0])*res:(i//coarse_grid[0]+1)*res+1, 
                                   (i%coarse_grid[1])*res:(i%coarse_grid[1]+1)*res+1]
        idx = glob_view.flatten()
        loc = np.vstack([2*idx, 2*idx+1]).T.flatten()
        K_idx = tuple(layout[i])
        K_glob[np.ix_(loc, loc)] += K[K_idx]
    
    return K_glob


def apply_BCs(K_global, F, fixed_dofs, inf=True):
    n_nodes = len(K_global)
    free_dofs = np.setdiff1d(np.arange(n_nodes), fixed_dofs)
    if inf==True:
        zero = np.where(~K_global.any(axis=1))[0]
        dof = np.setdiff1d(free_dofs, zero)
        K_ff = K_global[np.ix_(dof, dof)]
        F_f = F[dof]
        return K_ff, F_f, dof
    else:
        K_global[:, fixed_dofs] = 0
        K_global[fixed_dofs, :] = 0
        K_global[fixed_dofs, fixed_dofs] = 1
        F_f = F.copy()
        F_f[fixed_dofs] = 0
        return K_global, F_f, fixed_dofs


def generate_structure(grid, choices):
    # Create a grid filled with zeros (white pixels)
    structure = np.zeros((2, grid[0], grid[1]), dtype=int)

    # Randomly choose the number of pixels for the shape
    num_moves = np.random.randint(3, 2*grid[0]*grid[1])

    # Define possible moves (up, down, left, right)
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # Create a function to check if a move is valid
    def is_valid_move(x, y):
        return (0 <= x < grid[0]) and (0 <= y < grid[1])

    # Randomly choose a starting point for the shape
    current_x, current_y = np.random.randint(0, grid[0]), np.random.randint(0, grid[1])
    structure[:, current_x, current_y] = random.choice(choices)

    # Generate the shape
    for _ in range(num_moves):
        # Attempt to extend the shape
        move = random.choice(moves)
        new_x, new_y = current_x + move[0], current_y + move[1]
        if is_valid_move(new_x, new_y):
            current_x, current_y = new_x, new_y 
            structure[:, current_x, current_y] = random.choice(choices)

    return structure


def generate_support(grid, layout):
    # generate essential supports
    non_zero = np.where(layout.any(axis=1))[0]
    elements = []
    grid_num = np.arange((grid[0]+1)*(grid[1]+1)).reshape(grid[0]+1, grid[1]+1)
    for i in range(len(non_zero)):
        col = non_zero[i] % grid[0]
        row = non_zero[i] // grid[1]
        elements.append(grid_num[row:row+2, col:col+2].flatten())
    support_1 = random.sample(elements, 1)[0][random.randint(0, 3)]
    support_2 = support_1
    while support_2 == support_1:
        support_2 = random.sample(elements, 1)[0][random.randint(0, 3)]
    row = support_1 // (grid[1] + 1)
    col = support_1 % (grid[0] + 1)
    if (support_2 // (grid[1] + 1)) == row:
        x_or_y = 1
    elif (support_2 % (grid[0] + 1)) == col:
        x_or_y = 0
    else:
        x_or_y = random.randint(0, 1)

    supports = [2*support_1, 2*support_1+1, 2*support_2+x_or_y]
    n_support = random.randint(0, (len(elements)+1)//4)
    # generate additional supports
    i = 0
    while i < n_support:
        new_support = random.choice(elements)[random.randint(0, 3)]
        x_or_y = random.randint(0, 1)
        if 2*new_support+x_or_y not in supports:
            supports.append(2*new_support+x_or_y)
            i += 1
    return supports


def generate_support_bridge(grid, layout):
    # generate essential supports
    grid_num = np.arange((grid[0]+1)*(grid[1]+1)).reshape(grid[0]+1, grid[1]+1)
    non_zero = np.where(layout[0]!=0)
    
    most_left_modules = np.where(non_zero[1] == non_zero[1].min())[0]
    elements = []
    for i in most_left_modules:
        row = non_zero[0][i]
        col = non_zero[1][i]
        elements.append(grid_num[row:row+2, col].flatten())
    support_1 = random.sample(elements, 1)[0][random.randint(0, 1)]
    
    most_right_modules = np.where(non_zero[1] == non_zero[1].max())[0]
    elements = []
    for i in most_right_modules:
        row = non_zero[0][i]
        col = non_zero[1][i]
        elements.append(grid_num[row:row+2, col+1].flatten())
    support_2 = random.sample(elements, 1)[0][random.randint(0, 1)]

    supports = [2*support_1, 2*support_1+1, 2*support_2+1]
        
    return supports


def create_image_from_layout(layout_array, res=8):
    """
    Create an image by placing module images based on a layout array.
    
    :param layout_array: A (2, M, N) numpy array defining the layout of module images.
    :param res: Resolution of each module image (width and height in pixels).
    :return: A black and white image constructed from module images.
    """
          
    M, N = layout_array.shape[1], layout_array.shape[2]
    
    # Create an empty image in grayscale mode
    img_width, img_height = res * N, res * M
    final_image = Image.new('L', (img_width, img_height))  # 'L' for grayscale
    
    # Loop through each grid position
    for i in range(M):
        for j in range(N):
            # Construct the module filename
            module_file = f'modules_{res}/{layout_array[0, i, j]:.0f}_{layout_array[1, i, j]:.0f}.png'
                
            # Open the module image and convert it to grayscale
            module_image = Image.open(module_file).convert('L')
                
            # Calculate the position to paste the module image
            top_left_corner = (j * res, i * res)
                
            # Paste the module image into the final image
            final_image.paste(module_image, top_left_corner)
                
    return final_image


def create_image_3(layout_array, res=8):
    """
    Create an image by placing module images based on a layout array.
    
    :param layout_array: A (1, M, N) numpy array defining the layout of module images.
    :param res: Resolution of each module image (width and height in pixels).
    :return: A black and white image constructed from module images.
    """
          
    M, N = layout_array.shape[1], layout_array.shape[2]
    
    # Create an empty image in grayscale mode
    img_width, img_height = res * N, res * M
    final_image = Image.new('L', (img_width, img_height))  # 'L' for grayscale
    
    # Loop through each grid position
    for i in range(M):
        for j in range(N):
            # Construct the module filename
            module_file = f'modules_3/{layout_array[0, i, j]:.0f}.png'
                
            # Open the module image and convert it to grayscale
            module_image = Image.open(module_file).convert('L')
                
            # Calculate the position to paste the module image
            top_left_corner = (j * res, i * res)
                
            # Paste the module image into the final image
            final_image.paste(module_image, top_left_corner)
                
    return final_image


def find_deform(K, f):
    u = torch.linalg.solve(K, f)
    u_f = (u * f).sum(dim=(-2, -1))
    return u, u_f