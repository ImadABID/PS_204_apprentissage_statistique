function image_class = classify_gauss(image_to_classify, data_trn, lb_trn, x_bar, U, l, N, size_cls_trn, Nc)

    
    % --- w representation of image to classify
    image_to_classify_2_w = x2w(image_to_classify, x_bar, U, l);
    
    % --- w representation of training set
    data_trn_2_w = zeros(l, N);
    for i = 1:N
        data_trn_2_w(:, i) = x2w(data_trn(:, i), x_bar, U, l);
    end
    
    % --- computing mean of same face images
    same_face_mean_2_w = zeros(l, Nc);
    for class_ii=1:Nc
        class_indexes = unique(lb_trn);
        images_same_face = data_trn_2_w(:, find(lb_trn==class_indexes(class_ii)));
        same_face_mean_2_w(:, class_ii) = mean(images_same_face, 2);
    end
    
    % --- computing covariance matrix
    i = 1;
    sigma = zeros(l, l);
    for class_ii=1:Nc
        for image_i=1:size_cls_trn(class_ii)
            sigma = sigma + (data_trn_2_w(:, i) - same_face_mean_2_w(:,class_ii)) * (data_trn_2_w(:, i) - same_face_mean_2_w(:,class_ii))';
            i = i + 1;
        end
    end
    sigma = 1/N * sigma;
    
    
    % --- computing den
    tab=[];
    for j=1:Nc
        tab(j)= (image_to_classify_2_w - same_face_mean_2_w(:,j))' * inv(sigma) * (image_to_classify_2_w - same_face_mean_2_w(:,j));
    end
    % --- find k 
    [~, tmp]=min(tab);
    
    image_class = class_indexes(tmp);
  

end
