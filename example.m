image_dir = './images/';

fid = fopen('metadata.csv');
data = textscan(fid, '%s%f%f%f%f', 'delimiter', ',');
fclose(fid);

[images, left_x, left_y, right_x, right_y] = deal(data{:});

for ix = 1:numel(images)
  
  im = imread([image_dir images{ix}]);
  sz = size(im); sz = sz(1:2);
  
  figure(1); clf;
  image(im, 'XData', [1 sz(2)] - (sz(2)+1)/2, 'YData', [sz(1) 1] - (sz(1)+1)/2)
  axis xy image off
  hold on
  plot([left_x(ix) right_x(ix)], [left_y(ix) right_y(ix)], 'g', 'LineWidth', 3);
  hold off
  
  pause
  
end