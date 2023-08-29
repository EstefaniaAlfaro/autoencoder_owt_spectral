function rescale_data = rescale_data_interval(data_scale, range_value)
    number_end_members = size(data_scale, 1);
    rescale_data = zeros(size(data_scale));
    for index=1:number_end_members
        rescale_data(index, :) = rescale(data_scale(index, :), ...
            range_value(1), range_value(end));
    end
end