function w = x2w(x, x_bar, U, l)
    w=zeros(l,1);
    for i = 1:l
        w(i) = (x-x_bar)' * U(:,i);
    end
end
