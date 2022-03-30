function [data_trn, lb_trn, P, N, Nc, size_cls_trn] = data_extraction(adr)
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
end