%% ����Ѫ��������ʵѪ��������ά����Ѫ���ܶȵĶԱ�
clear;clc;

%% Read Image and Intialization
VesType = "art Trees";

Net_list = ["389","546","913"];

tree_number=3; % ÿ�����ͣ���������Ķ���
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
disp(strcat("=====Ѫ�����ͣ�",VesType,"====="))
disp(strcat("ԭʼ��������ά����",num2str(mean(FD(1,:))),"��",num2str(std(FD(1,:))),...
    " Ѫ���ܶȣ�",num2str(mean(VD(1,:))),"��",num2str(std(VD(1,:)))))


%% �м��������������ά��
function [D,X,Y] = BCD(img)
%% ����ͨ��FracLab����㷨����Box-countingά�������Եõ���ɢ��������Զȼ��
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
%% Size��Χѡ�񣬴���FracLab��ĺ������м���
Min = -floor(log2(min(size(img))));
Max = -1;
Size = 2.^(Min:0.5:Max); % �̶����
% Size = 2.^linspace(Min,Max,15); % �̶�����

reg = 0;  % ��ȫ�߶Ƚ�����ϣ�ͬʱ����ʾͼ��
Waitbar = 0;  % �����ּ�ʱ��

% [boxdim,Nboxes,handlefig]= boxdim_binaire(BWimg, Size, Ratio, Axes, Waitbar ,reg, lrstr, optvarargin)
[D,NBoxes] = boxdim_binaire(img,Size,[],[],Waitbar,reg); % fraclab����

X = log(min(size(img)).*Size); 
Y = log(NBoxes); 
end
