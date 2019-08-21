function [ coordinate ] = getLane( score )
% Calculate lane position from a probmap.
thr = 0.3;
coordinate = zeros(1,18);
for i=1:18
    lineId = uint16(208-(i-1)*20/350*208);
    line = score(lineId,:);
    [value, id] = max(line);
    if double(value)/255 > thr
        coordinate(i) = id;
    end
end
if sum(coordinate>0)<2
    coordinate = zeros(1,18);
end
end
