% P. Vallet (Bordeaux INP), 2019

clc;
clear all;
close all;

%% Data extraction
% Training set
adr = './database/training2/';
fld = dir(adr);
nb_elt = length(fld);
% Data matrix containing the training images in its columns 
data_trn = []; 
% Vector containing the class of each training image
lb_trn = []; 
for i=1:nb_elt
    if fld(i).isdir == false
        lb_trn = [lb_trn ; str2num(fld(i).name(6:7))];
        img = double(imread([adr fld(i).name]));
        data_trn = [data_trn img(:)];
    end
end
% Size of the training set
[P,N] = size(data_trn);
% Classes contained in the training set
[lb_trn,I]=sort(lb_trn);
data_trn = data_trn(:,I);
[cls_trn,bd,~] = unique(lb_trn);
Nc = length(cls_trn); 
% Number of training images in each class
size_cls_trn = [bd(2:Nc)-bd(1:Nc-1);N-bd(Nc)+1]; 

%% Display the database
F = zeros(192*Nc,168*max(size_cls_trn));
for i=1:Nc
    for j=1:size_cls_trn(i)
          pos = sum(size_cls_trn(1:i-1))+j;
          F(192*(i-1)+1:192*i,168*(j-1)+1:168*j) = reshape(data_trn(:,pos),[192,168]);
    end
end
figure;
imagesc(F);
colormap(gray);
axis off;

%% Réduction de dimension

% --- calcule des vecteurs propres

x_bar = mean(data_trn, 2);

X = 1/sqrt(N) * (data_trn-x_bar);

Gram = (X')*X;
[V, D] = eig(Gram);

% --- --- elimination de v associé à 0
% DD = ((D==0) + 10)';
% [~, vec_0_colum_index] = min(DD(:));
% vec_0_colum_index = mod(vec_0_colum_index-1, N)+1;

[~, Index_order] = sort(diag(D));

V = V(:, Index_order);
V = V(:,2:end);
V = V(:,end:-1:1);

U = X*V * ((V')*(X')*X*V)^(-1/2);

U = [U zeros(P, 1)];

% --- calcule des valeurs propres

U_val = zeros(1,N-1);
for i=1:1:N-1
    lmbda_u = X * (X'*U(:,i));
    [non_null_val, non_null_index] = max(lmbda_u);
    U_val(1, i) = non_null_val / U(non_null_index,i);
end

%% kk ration
alpha = 0.9;
L = -1;
kk = zeros(1, N-1);
for l=1:1:N-1
    kk(1, l) = sum(U_val(1:l)) / sum(U_val);
    if L == -1 && kk(1, l) >= alpha
        L = l;
    end
end

figure, plot((1:N-1), kk);

%% Display U

figure,
size_cls_trn_max = max(size_cls_trn);
for i=1:Nc
    for j=1:size_cls_trn_max
        subplot(Nc,size_cls_trn_max, (i-1)*size_cls_trn_max+j);
        imagesc(reshape(U(:, (i-1)*size_cls_trn_max+j), [192,168]));
        colormap(gray);
    end
end

%% reconstruction
figure,
for i=1:Nc
    for j=1:size_cls_trn_max
        image_index = (i-1)*size_cls_trn_max+j;
        x = data_trn(:, image_index) - x_bar;
        x_acp = zeros(size(x)) ;
        for l=1:1:L
            x_acp = x_acp + (x' * U(:, l)) * U(:, l);
        end
        subplot(Nc,size_cls_trn_max, image_index);
        imagesc(reshape(x_acp+x_bar, [192,168]));
        colormap(gray);
    end
end

%% l*
fprintf("la dimension l* du sous-espace de reconstruction de telle manière à garantir un ratio de %f est %d.\n", alpha, L);

%% Classifieur k-NN

% --- classification params
image_to_classify_path = "./database/test1/yaleB01_P00A+005E+10.pgm";
k = 12;

% --- read & w representation of image to classify
image_to_classify = imread(image_to_classify_path);
image_to_classify = double(image_to_classify(:));
image_to_classify_2_w = x2w(image_to_classify, x_bar, U, l);

% --- w representation of training set
data_trn_2_w = zeros(size(data_trn));
for i = 1:N
    data_trn_2_w(:, i) = x2w(data_trn(:, i), x_bar, U, l);
end

% --- find k-Nearest Neighbors
k_NN_indexes = zeros(1,k);

for k_index = 1:k
    min_dist = -1;
    min_dist_index = -1;
    for neighbors_index = 1:N
        dist = norm(image_to_classify_2_w - data_trn_2_w(:, neighbors_index));
        fprintf("dist = %f\n", dist);
        if ...
                ~val_in_table(k_NN_indexes(1:k_index-1), neighbors_index) && ...
                (min_dist_index == -1 || min_dist > dist)
        
            % --------
            min_dist_index = neighbors_index;
            min_dist = dist;
        end
    end
    k_NN_indexes(1, k_index) = min_dist_index;
end

close all;

