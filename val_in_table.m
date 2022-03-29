function y = val_in_table(tab,x)
    y = sum(tab==x) > 0;
end
