%% 生成血管树与真实血管树分形维数与血管密度的对比
clear;clc;

%% Read Image and Intialization
VesType = "art Trees";

Net_list = ["389","546","913"];

tree_number=3; % 每个类型，输入的树的多少
FD = zeros(2,length(Net_list)*tree_number);
VD = zeros(2,length(Net_list)*tree_number);

for i = 1:length(Net_list)
    for j = 1:tree_number
        img_path = strcat('..\Data\',VesType,'\Tree_',Net_list(i),'_',num2str(j-1),'.tiff');
        disp(strcat('..\Data\',VesType,'\ori_',Net_list(i),'_',num2str(j-1),'.tiff'))
        img_tmp = imread(img_path);
        grayimg = img_tmp(:,:,1);

        th = graythresh(grayimg);
        img = double(~imbinarize(grayimg,th));         

       %% Calculate Lacunarity
        [FD(1,(i-1)*tree_number+j),~,~] = BCD(img);
        VD(1,(i-1)*tree_number+j) = sum(sum(img))/(size(img,1)*size(img,2));

    end
end

disp(strcat('FD:',num2str(FD(1,:))))
disp(strcat('VD:',num2str(VD(1,:))))
disp(strcat("=====血管类型：",VesType,"====="))
disp(strcat("原始树：分形维数：",num2str(mean(FD(1,:))),"±",num2str(std(FD(1,:))),...
    " 血管密度：",num2str(mean(VD(1,:))),"±",num2str(std(VD(1,:)))))


%% 盒计数方法计算分形维数
function [D,X,Y] = BCD(img)
%% 函数通过FracLab里的算法计算Box-counting维数，并对得到的散点进行线性度检测
% reg     : 0 : All box sizes are taken into consideration when computing 
%               the linear regression.
%           1 : The output 'dim' is empty, but a figure appears.
%               In this figure, you will be able to choose manually a 
%               range of sizes where the evolution of x versus y
%               is linear and to read the result of the calculation with this range.
%           2  : All box sizes are taken into consideration when computing 
%               the output 'dim'. Moreover, the same figure appears on
%               which you will be able to choose another regression range.
%           -1 : The same figure appears, but the function waits for your choice
%               of bounds and returns the resulting dimension in the
%               output 'dim'.
%% Size范围选择，带入FracLab里的函数进行计算
Min = -floor(log2(min(size(img))));
Max = -1;
Size = 2.^(Min:0.5:Max); % 固定间距
% Size = 2.^linspace(Min,Max,15); % 固定点数

reg = 0;  % 对全尺度进行拟合，同时不显示图形
Waitbar = 0;  % 不出现计时框

% [boxdim,Nboxes,handlefig]= boxdim_binaire(BWimg, Size, Ratio, Axes, Waitbar ,reg, lrstr, optvarargin)
[D,NBoxes] = boxdim_binaire(img,Size,[],[],Waitbar,reg); % fraclab函数

X = log(min(size(img)).*Size); 
Y = log(NBoxes); 
end
