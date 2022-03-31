function image_class = classify_k_NN(image_to_classify, data_trn, lb_trn, x_bar, U, l, N)
    % --- params
    k = 12;
    
    % --- w representation of image to classify
    image_to_classify_2_w = x2w(image_to_classify, x_bar, U, l);

    % --- w representation of training set
    data_trn_2_w = zeros(l,N);
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

    % --- find most dominante Neighbor
    [nbr_of_occurence_of_each_class, image_class_tab] = groupcounts(lb_trn(k_NN_indexes)');
    [~, image_class_index] = max(nbr_of_occurence_of_each_class);
    image_class = image_class_tab{image_class_index};

end
