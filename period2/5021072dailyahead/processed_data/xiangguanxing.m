x = csvread('x_m.csv');
y = csvread('y_m.csv');
total_hours = length(y);
x = x(1: total_hours - 24, :);
y = y(25: total_hours,: );
data = [y x];
rr = [];
for i = 1: length(data(1, :))
    mean_value = mean(data(:, i));
    std_value = std(data(:, i));
    data(:, i) = (data(:, i) - mean_value) / std_value;
end

for i = 1: (length(data(1, :)) - 1)
    rr(1, i) = corr(data(:, 1), data(:, i+1), 'type', 'Pearson');
end

index = [];
n = length(x(:, 1));
for i = 1: length(rr)
    if(abs(rr(i)) >= 0.3)
        index(:, i) = ones(n, 1);
        i
    else
        index(:, i) = zeros(n, 1);
    end
end
x_new = x .* index;
csvwrite('x_rr.csv', x_new);

% count = 1;
% index = [];
% for i = 1: 135
%     if(abs(rr(1, i)) > 0.1)
%         index(count) = i;
%         count = count + 1;
%     end
% end
% for i=1:length(rr)-4
%     if(abs(rr(i)) < 0.1)
%         rr(i) = 0;
%     end
% end
% rr(length(rr)-3:length(rr)) = [0 0 0 0];
% for i=1:463
%     p(i, 1:3) = polyfit(year, eco_affectoi(i, :), 2);
% end
% for i=1:463
%     fore_year = [7 8 9 10 11];
%     fore_eco(i, 1:5) = polyval(p(i, 1:3), fore_year);
% end
