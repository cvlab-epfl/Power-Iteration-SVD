% zca test
%% generate eclipse sample points
x1 = 2; y1 = 4;
x2 = 7; y2 = 8;
e = 0.999;
a = 1/2*sqrt((x2-x1)^2+(y2-y1)^2);
b = a*sqrt(1-e^2);
t = linspace(0,2*pi);
X = a*cos(t);
Y = b*sin(t);
w = atan2(y2-y1,x2-x1);
x = (x1+x2)/2 + X*cos(w) - Y*sin(w);
y = (y1+y2)/2 + X*sin(w) + Y*cos(w);
%  plot(x,y,'-')
%  hold on;
%  plot(x,y,'*')
%  axis equal
 
%% adding noise
noisex = rand(size(x))*0.6-0.3;
noisey = rand(size(y))*0.6-0.3;
x = x+noisex;
y = y+noisey;

%% standardize data
xcenter = mean(x);
ycenter = mean(y);
xstd = std(x);
ystd = std(y);
x = x - xcenter;
y = y - ycenter;
x = x/xstd;
y = y/ystd;

% visualize standardized data
plot(x,y,'*')
hold on;
axis equal

%% eigen vectors
points = [x(:),y(:)];
[m, n] = size(points);
M = points'*points;
power_iter = 10;
eigen_vectors = cell(0);
lambdas = cell(0);
for i = 1:n
    v = ones(n,1);
    for j = 1:power_iter
        v = M*v/sqrt((M*v)'*(M*v));
    end
    lambda_ = M*v./v;
    lambdas{end+1} = mean(lambda_);
    M = M - (M*v)*v';
    eigen_vectors{end+1} = v;
end

% visualize eigenvectors
for i = 1:length(eigen_vectors)
    vector = (eigen_vectors{i})*2;
    xo = vector(1); yo = vector(2);
    plot([0, xo], [0,yo]);
end

%% reconstruction
xr = zeros(size(points));
for i = 1: length(eigen_vectors)
    vector = eigen_vectors{i};
    lambda_ = lambdas{i};
    xr = xr + (points*vector/sqrt(lambda_))*vector';
end

plot(xr(:,1),xr(:,2),'ro');
