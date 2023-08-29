clear all;
close all;
clc;
load('results_samson.mat')
range_value = [0, 1];
colors = {'r', 'g', 'b'};
rescale_data_nsae = rescale_data_interval(endmembers_NSAE_samson, range_value);
rescale_data_undip = rescale_data_interval(end_members_samson_undip.', ...
   range_value);
rescale_data_cnna = rescale_data_interval(end_members_cnnae, range_value);
rescale_data_gt = rescale_data_interval(M.', range_value);
for index=1:size(rescale_data_nsae, 1)
    plot(rescale_data_gt(index, :), colors{index}, 'LineWidth', 1.7)
    hold on;
    grid on;
end
xlabel('Bands')
ylabel('Reflectance')
legend("Soil", "Tree" ,"Water")
%%
% Plot the water
figure,
plot(rescale_data_nsae(2, :), 'b', 'LineWidth', 1.9)
hold on
plot(rescale_data_undip(1, :), 'g', 'LineWidth', 1.9)
hold on
plot(rescale_data_cnna(2, :), 'r', 'LineWidth', 1.9)
hold on;
plot(rescale_data_gt(3, :), 'c', 'LineWidth', 1.9)
grid on;
legend('NSAE-SU', 'UnDIP', 'CNNAEU', 'GT', 'FontSize', 12)
xlabel('Number of Bands', 'FontSize', 18);
ylabel('Reflectance', 'FontSize', 18);
set(gcf,'Position',[200 200 800 600])
%%
% Plot soil
figure,
plot(rescale_data_nsae(1, :), 'b', 'LineWidth', 1.9)
hold on
plot(rescale_data_undip(2, :), 'g', 'LineWidth', 1.9)
hold on
plot(rescale_data_cnna(1, :), 'r', 'LineWidth', 1.9)
hold on;
plot(rescale_data_gt(1, :), 'c', 'LineWidth', 1.9)
grid on;
legend('NSAE-SU', 'UnDIP', 'CNNAEU', 'GT', 'FontSize', 12)
xlabel('Number of Bands', 'FontSize', 18);
ylabel('Reflectance', 'FontSize', 18);
set(gcf,'Position',[300 200 800 600])
%%
% Plot Tree
figure,
plot(rescale_data_nsae(3, :), 'b', 'LineWidth', 1.9)
hold on
plot(rescale_data_undip(3, :), 'g', 'LineWidth', 1.9)
hold on
plot(rescale_data_cnna(3, :), 'r', 'LineWidth', 1.9)
hold on;
plot(rescale_data_gt(2, :), 'c', 'LineWidth', 1.9)
grid on;
legend('NSAE-SU', 'UnDIP', 'CNNAEU', 'GT', 'FontSize', 12)
xlabel('Number of Bands', 'FontSize', 18);
ylabel('Reflectance', 'FontSize', 18);
set(gcf,'Position',[300 200 800 600])
%%
%%
labels_gt = {3, 1, 2};
labels_nsae = {2, 1, 3};
num_end_members = size(rescale_data_nsae, 1);
score = zeros(num_end_members, 1);
for index_score = 1:num_end_members
    score(index_score) = sam(rescale_data_nsae(labels_nsae{index_score}, :), ...
        rescale_data_gt(labels_gt{index_score}, :));
end
