%%%%% Create the cuboid domain %%%%%
N = 13; % number of nodes in each direction including the border
[x1,x2,x3] = meshgrid(linspace(0,2*pi,N));
x1 = x1(:);
x2 = x2(:);
x3 = x3(:);
triang = delaunayTriangulation(x1,x2,x3); % create tetrahedral elements

%%%%% Find the faces of the cuboid %%%%%
f1 = find(x3 == 0);
f2 = find(x3 == 2*pi);
f3 = find(x2 == 0);
f4 = find(x2 == 2*pi);
f5 = find(x1 == 0);
f6 = find(x1 == 2*pi);

%%%%% Define matrix A(x) %%%%%
A_matrix = @(x1, x2, x3) [ 7+sign(sin(1.5*x1)*sin(1.5*x2)), -2-sign(sin(1.5*x2)*sin(1.5*x3)),   sign(sin(1.5*x1)*sin(1.5*x2)*sin(1.5*x3));
                          -2-sign(sin(1.5*x2)*sin(1.5*x3)), 4.01+sign(sin(1.5*x1)*sin(1.5*x2)), 0;
                          sign(sin(1.5*x1)*sin(1.5*x2)*sin(1.5*x3)), 0, 3+sign(sin(1.5*x2)*sin(1.5*x3))];

%%%%% Define basis functions xi_i %%%%%
xi = {[1; 0; 0], [0; 1; 0], [0; 0; 1]}; 

%%%%% Construct global stiffness matrix K %%%%%
n_elem = length(triang.ConnectivityList);
n_nodes = length(triang.Points);
%K = sparse(n_nodes,n_nodes);
K_i = [];
K_j = [];
K_v = [];
%K_inv_1 = sparse(2*n_nodes,2*n_nodes);
K_inv_1_i = [];
K_inv_1_j = [];
K_inv_1_v = [];
%K_inv_2 = sparse(3*n_nodes,3*n_nodes);
K_inv_2_i = [];
K_inv_2_j = [];
K_inv_2_v = [];

f = {zeros(n_nodes,1), zeros(n_nodes,1), zeros(n_nodes,1)};
f_1 = {zeros(2*n_nodes,1), zeros(2*n_nodes,1), zeros(2*n_nodes,1)};
f_2 = {zeros(3*n_nodes,1), zeros(3*n_nodes,1), zeros(3*n_nodes,1)};

for i = 1:n_elem
    elem = triang.ConnectivityList(i,:);
    node_coords = triang.Points(elem,:);
    vectors = node_coords(2:end,:)-node_coords(1:end-1,:);
    volume = (1/6)*abs(dot(vectors(1,:), cross(vectors(2,:), vectors(3,:))));
    center = mean(node_coords, 1);
    M_inv = inv([ones(4,1), node_coords]);
    D_grad = M_inv(2:4,:);
    D_curl_1 = [-D_grad(2,:), -D_grad(3,:); 
                 D_grad(1,:),  zeros(1,4);
                 zeros(1,4),   D_grad(1,:)];
    D_curl_2 = [zeros(1,4),  D_grad(3,:),-D_grad(2,:); 
               -D_grad(3,:), zeros(1,4),  D_grad(1,:);
                D_grad(2,:),-D_grad(1,:), zeros(1,4)];
    A = A_matrix(center(1),center(2),center(3));
    A_inv = inv(A);
    K_elem = volume * D_grad' * A * D_grad;
    K_inv_elem_1 = volume * D_curl_1' * A_inv * D_curl_1;
    K_inv_elem_2 = volume * D_curl_2' * A_inv * D_curl_2;
    %K(elem,elem) = K(elem,elem) + K_elem;
    ij = repmat(elem, 4, 1);
    K_i(end+1:end+4*4) = ij(:);
    ij = ij';
    K_j(end+1:end+4*4) = ij(:);
    K_v(end+1:end+4*4) = K_elem(:);
    %K_inv_1([elem, N^3+elem],[elem, N^3+elem]) = K_inv_1([elem, N^3+elem],[elem, N^3+elem]) + K_inv_elem_1;
    ij = repmat([elem, N^3+elem], 8, 1);
    K_inv_1_i(end+1:end+8*8) = ij(:);
    ij = ij';
    K_inv_1_j(end+1:end+8*8) = ij(:);
    K_inv_1_v(end+1:end+8*8) = K_inv_elem_1(:);
    %K_inv_2([elem, N^3+elem, 2*N^3+elem],[elem, N^3+elem, 2*N^3+elem]) = K_inv_2([elem, N^3+elem, 2*N^3+elem],[elem, N^3+elem, 2*N^3+elem]) + K_inv_elem_2;
    ij = repmat([elem, N^3+elem, 2*N^3+elem], 12, 1);
    K_inv_2_i(end+1:end+12*12) = ij(:);
    ij = ij';
    K_inv_2_j(end+1:end+12*12) = ij(:);
    K_inv_2_v(end+1:end+12*12) = K_inv_elem_2(:);
    for j = 1:3
        f_elem = - volume * D_grad' * A * xi{j};
        f_1_elem = - volume * D_curl_1' * A_inv * xi{j};
        f_2_elem = - volume * D_curl_2' * A_inv * xi{j};
        f{j}(elem) = f{j}(elem) + f_elem;
        f_1{j}([elem, N^3+elem]) = f_1{j}([elem, N^3+elem]) + f_1_elem;
        f_2{j}([elem, N^3+elem, 2*N^3+elem]) = f_2{j}([elem, N^3+elem, 2*N^3+elem]) + f_2_elem;
    end
end
K = sparse(K_i, K_j, K_v, n_nodes, n_nodes);
K_inv_1 = sparse(K_inv_1_i, K_inv_1_j, K_inv_1_v, 2*n_nodes, 2*n_nodes);
K_inv_2 = sparse(K_inv_2_i, K_inv_2_j, K_inv_2_v, 3*n_nodes, 3*n_nodes);

% find unique nodes belonging to the master and copy faces in the periodic BCs
master_faces = unique([f1; f3; f5], 'stable'); 
master_faces = setdiff(master_faces, f2, 'stable');
master_faces = setdiff(master_faces, f4, 'stable');
master_faces = setdiff(master_faces, f6, 'stable');
copy_faces = unique([f2; f4; f6], 'stable');
copy_faces = setdiff(copy_faces, master_faces, 'stable');
free_nodes = setdiff(1:n_nodes, copy_faces, 'stable');
n_master = length(master_faces);
n_free = length(free_nodes);

%%%%% Apply periodic boundary conditions to the global stiffness matrix%%%%%
K_copy = K;
K_copy(:,f1) = K_copy(:,f1) + K_copy(:,f2);
K_copy(f1,:) = K_copy(f1,:) + K_copy(f2,:);
K_copy(:,f3) = K_copy(:,f3) + K_copy(:,f4);
K_copy(f3,:) = K_copy(f3,:) + K_copy(f4,:);
K_copy(:,f5) = K_copy(:,f5) + K_copy(:,f6);
K_copy(f5,:) = K_copy(f5,:) + K_copy(f6,:);
K_per = K_copy(free_nodes, free_nodes);

%%%%% Apply periodic boundary conditions to the inverse global stiffness (compliance) matrix%%%%%
K_inv_copy_1 = K_inv_1;
K_inv_copy_1(:,[f1, N^3+f1]) = K_inv_copy_1(:,[f1, N^3+f1]) + K_inv_copy_1(:,[f2, N^3+f2]);
K_inv_copy_1([f1, N^3+f1],:) = K_inv_copy_1([f1, N^3+f1],:) + K_inv_copy_1([f2, N^3+f2],:);
K_inv_copy_1(:,[f3, N^3+f3]) = K_inv_copy_1(:,[f3, N^3+f3]) + K_inv_copy_1(:,[f4, N^3+f4]);
K_inv_copy_1([f3, N^3+f3],:) = K_inv_copy_1([f3, N^3+f3],:) + K_inv_copy_1([f4, N^3+f4],:);
K_inv_copy_1(:,[f5, N^3+f5]) = K_inv_copy_1(:,[f5, N^3+f5]) + K_inv_copy_1(:,[f6, N^3+f6]);
K_inv_copy_1([f5, N^3+f5],:) = K_inv_copy_1([f5, N^3+f5],:) + K_inv_copy_1([f6, N^3+f6],:);
K_inv_per_1 = K_inv_copy_1([free_nodes, N^3+free_nodes], [free_nodes, N^3+free_nodes]);

K_inv_copy_2 = K_inv_2;
K_inv_copy_2(:,[f1, N^3+f1, 2*N^3+f1]) = K_inv_copy_2(:,[f1, N^3+f1, 2*N^3+f1]) + K_inv_copy_2(:,[f2, N^3+f2, 2*N^3+f2]);
K_inv_copy_2([f1, N^3+f1, 2*N^3+f1],:) = K_inv_copy_2([f1, N^3+f1, 2*N^3+f1],:) + K_inv_copy_2([f2, N^3+f2, 2*N^3+f2],:);
K_inv_copy_2(:,[f3, N^3+f3, 2*N^3+f3]) = K_inv_copy_2(:,[f3, N^3+f3, 2*N^3+f3]) + K_inv_copy_2(:,[f4, N^3+f4, 2*N^3+f4]);
K_inv_copy_2([f3, N^3+f3, 2*N^3+f3],:) = K_inv_copy_2([f3, N^3+f3, 2*N^3+f3],:) + K_inv_copy_2([f4, N^3+f4, 2*N^3+f4],:);
K_inv_copy_2(:,[f5, N^3+f5, 2*N^3+f5]) = K_inv_copy_2(:,[f5, N^3+f5, 2*N^3+f5]) + K_inv_copy_2(:,[f6, N^3+f6, 2*N^3+f6]);
K_inv_copy_2([f5, N^3+f5, 2*N^3+f5],:) = K_inv_copy_2([f5, N^3+f5, 2*N^3+f5],:) + K_inv_copy_2([f6, N^3+f6, 2*N^3+f6],:);
K_inv_per_2 = K_inv_copy_2([free_nodes, N^3+free_nodes, 2*N^3+free_nodes], [free_nodes, N^3+free_nodes, 2*N^3+free_nodes]);

%%%%% Construct the right-hand vectors%%%%%
f_copy = f;
f_copy_1 = f_1;
f_copy_2 = f_2;
for j = 1:3
    f_copy{j}(f1) = f_copy{j}(f1) + f_copy{j}(f2);
    f_copy{j}(f3) = f_copy{j}(f3) + f_copy{j}(f4);
    f_copy{j}(f5) = f_copy{j}(f5) + f_copy{j}(f6);
    
    f_copy_1{j}([f1, N^3+f1]) = f_copy_1{j}([f1, N^3+f1]) + f_copy_1{j}([f2, N^3+f2]);
    f_copy_1{j}([f3, N^3+f3]) = f_copy_1{j}([f3, N^3+f3]) + f_copy_1{j}([f4, N^3+f4]);
    f_copy_1{j}([f5, N^3+f5]) = f_copy_1{j}([f5, N^3+f5]) + f_copy_1{j}([f6, N^3+f6]);
    
    f_copy_2{j}([f1, N^3+f1, 2*N^3+f1]) = f_copy_2{j}([f1, N^3+f1, 2*N^3+f1]) + f_copy_2{j}([f2, N^3+f2, 2*N^3+f2]);
    f_copy_2{j}([f3, N^3+f3, 2*N^3+f3]) = f_copy_2{j}([f3, N^3+f3, 2*N^3+f3]) + f_copy_2{j}([f4, N^3+f4, 2*N^3+f4]);
    f_copy_2{j}([f5, N^3+f5, 2*N^3+f5]) = f_copy_2{j}([f5, N^3+f5, 2*N^3+f5]) + f_copy_2{j}([f6, N^3+f6, 2*N^3+f6]);
end

for j = 1:3
    f_per{j} = f_copy{j}(free_nodes);
    f_per_1{j} = f_copy_1{j}([free_nodes, N^3+free_nodes]);
    f_per_2{j} = f_copy_2{j}([free_nodes, N^3+free_nodes, 2*N^3+free_nodes]);
end

%%%%% Solve with the conjugate gradients %%%%%
for j = 1:3
    U_per{j} = pcg(K_per, f_per{j}, [], length(f_per{j}));
    W_per_1{j} = pcg(K_inv_per_1, f_per_1{j}, [], length(f_per_1{j}));
    W_per_2{j} = pcg(K_inv_per_2, f_per_2{j}, [], length(f_per_2{j}));
end
U = {zeros(n_nodes,1), zeros(n_nodes,1), zeros(n_nodes,1)};
W_1 = {zeros(2*n_nodes,1), zeros(2*n_nodes,1), zeros(2*n_nodes,1)};
W_2 = {zeros(3*n_nodes,1), zeros(3*n_nodes,1), zeros(3*n_nodes,1)};
for j = 1:3
    U{j}(free_nodes) = U_per{j};
    U{j}(f2) = U{j}(f1);
    U{j}(f4) = U{j}(f3);
    U{j}(f6) = U{j}(f5);
    
    W_1{j}([free_nodes, N^3+free_nodes]) = W_per_1{j};
    W_1{j}([f2, N^3+f2]) = W_1{j}([f1, N^3+f1]);
    W_1{j}([f4, N^3+f4]) = W_1{j}([f3, N^3+f3]);
    W_1{j}([f6, N^3+f6]) = W_1{j}([f5, N^3+f5]);
    
    W_2{j}([free_nodes, N^3+free_nodes, 2*N^3+free_nodes]) = W_per_2{j};
    W_2{j}([f2, N^3+f2, 2*N^3+f2]) = W_2{j}([f1, N^3+f1, 2*N^3+f1]);
    W_2{j}([f4, N^3+f4, 2*N^3+f4]) = W_2{j}([f3, N^3+f3, 2*N^3+f4]);
    W_2{j}([f6, N^3+f6, 2*N^3+f6]) = W_2{j}([f5, N^3+f5, 2*N^3+f5]);
end

%%%%% Draw the solution %%%%%
figure()
for j=1:3
    subplot(1,3,j)
    slice(reshape(x1,N,N,N),reshape(x2,N,N,N),reshape(x3,N,N,N),reshape(U{j},N,N,N),[0, 2*pi],[0, 2*pi],[0, 2*pi]);
    shading interp
    colorbar()
    colormap jet
    camlight
    pbaspect([1 1 1])
end

%%%%% Compute the A_h, B_h1, B_h2 - upper and lower bounds to the homogenized parameters %%%%%
A_h = zeros(3,3);
B_h_1 = zeros(3,3);
B_h_2 = zeros(3,3);
for i = 1:n_elem
    elem = triang.ConnectivityList(i,:);
    node_coords = triang.Points(elem,:);
    vectors = node_coords(2:end,:)-node_coords(1:end-1,:);
    volume = (1/6)*abs(dot(vectors(1,:), cross(vectors(2,:), vectors(3,:))));
    center = mean(node_coords, 1);
    A = A_matrix(center(1),center(2),center(3));
    A_inv = inv(A);
    M_inv = inv([ones(4,1), node_coords]);
    D_grad = M_inv(2:4,:);
    D_curl_1 = [-D_grad(2,:), -D_grad(3,:); 
                 D_grad(1,:),  zeros(1,4);
                 zeros(1,4),   D_grad(1,:)];
    D_curl_2 = [zeros(1,4),    D_grad(3,:), -D_grad(2,:); 
                -D_grad(3,:),  zeros(1,4),   D_grad(1,:);
                D_grad(2,:),  -D_grad(1,:),  zeros(1,4)];
    for k = 1:3
       g_elem(:, k) = D_grad * U{k}(elem) + xi{k};  
       curl_1(:, k) = D_curl_1 * W_1{k}([elem, N^3+elem]) + xi{k};
       curl_2(:, k) = D_curl_2 * W_2{k}([elem, N^3+elem, 2*N^3+elem]) + xi{k};
    end
    A_h_elem = volume * g_elem' * A * g_elem;
    A_h = A_h + A_h_elem;
    B_h_elem_1 = volume * curl_1' * A_inv * curl_1;
    B_h_1 = B_h_1 + B_h_elem_1;
    B_h_elem_2 = volume * curl_2' * A_inv * curl_2;
    B_h_2 = B_h_2 + B_h_elem_2;
end
A_h = A_h / (8 * pi^3);
B_h_1 = B_h_1 / (8 * pi^3);
B_h_2 = B_h_2 / (8 * pi^3);

writematrix(N,'matrices.txt','WriteMode','append')
writematrix(A_h,'matrices.txt','WriteMode','append')
writematrix(B_h_1,'matrices.txt','WriteMode','append')
writematrix(B_h_2,'matrices.txt','WriteMode','append')

disp(inv(B_h_1))
disp(inv(B_h_2))
disp(A_h)