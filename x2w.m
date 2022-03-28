function w = x2w(x, x_bar, U, l)
    w=zeros(size(x));
    for i = 1:l
        w(i) = (x-x_bar)' * U(:,i);
    end
end
