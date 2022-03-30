% P. Vallet (Bordeaux INP), 2019

clc;
clear all;
close all;

%% Data extraction
% Training set
adr = './database/training1/';
[data_trn, lb_trn, P, N, Nc, size_cls_trn] = data_extraction(adr);

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
image_to_classify_path = "./database/test6/yaleB26_P00A+110E+65.pgm";

image_class = classify_k_NN(image_to_classify_path, data_trn, lb_trn, x_bar, U, l, N);


fprintf("image class = %d\n", image_class);

