clear all;
close all;
clc,
load('samson_1.mat');
load('end3.mat');
dataset_transpose = V'; 
dataset_reshape = reshape(dataset_transpose, nRow, nCol, nBand);
imagesc(dataset_reshape(:, :, 1))
colormap parula;
colors = {'-r', '-b', '-g'};
for index=1:size(M,2)
    plot(M(:, index), colors{index}, 'LineWidth', 1.7)
    hold on;
end
grid on;
title('Materials');
xlabel('Number of bands', 'FontWeight','bold');
ylabel('Reflectance', 'FontWeight','bold');
legend('Soil', 'Three', 'Water')
title_materials = {'Soil', 'Tree', 'Water'};
abundances_reshape = reshape(A', nRow, nCol, []);
figure
for index=1:size(abundances_reshape, 3)
    imagesc(abundances_reshape(:, :, index));
    title(title_materials{index})
    colorbar
    figure,
end
%%
load('results_samson.mat')
title_materials = {'Soil', 'Water', 'Tree'};
for index=1:size(abundances_NSAE_samson, 3)
    imagesc(abundances_NSAE_samson(:, :, index).');
    title(title_materials{index})
    colorbar
    figure,
end
%%
labels_gt = {1, 2, 3};
labels_nsae = {1,3,2};
rmse_abundances = zeros(size(abundances_NSAE_samson, 3),1);
for index_ab=1:size(abundances_NSAE_samson, 3)
    mean_square_error = immse(abundances_NSAE_samson(:, :, labels_nsae{index_ab}).', ...
        abundances_reshape(:, :, labels_gt{index_ab}));
    rmse_abundances(index_ab) = sqrt(mean_square_error);
end

