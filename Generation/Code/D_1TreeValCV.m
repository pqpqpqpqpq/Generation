%% ����Ѫ����k��lambda�����
%������������K lambda�Լ�angel��CV������
clear;clc;

%%
Net_list = ["389","546","913"];
% ѡ��Ҫ������ļ��к�����ļ���
% modeltype = ["art Trees"]; %����GAN��ģ�ͺͻ��ڷ�������ģ��
modeltype=["ven Trees"];
%%
%��ȥ��������ѭ��
rootpath = strcat("../Data/",modeltype); % ���ֶ����;���
k_cv = [];
lambda_cv = [];
angle_cv = [];

k_mean=[];
lambda_mean=[];
angle_mean=[];
all_mean_std=[];

for i = 1:length(Net_list)

   %% ����Ѫ����
     for j = 0:2

        filename = strcat('Tree_',Net_list(i),'_',num2str(j),'.mat');
        Mat = load(fullfile(rootpath,filename));
        Tree = Mat.data;

        filename = strcat('Tree_',Net_list(i),'_',num2str(j),'_Coord.mat');
        Mat = load(fullfile(rootpath,filename));
        Coord = Mat.data;

        k_mat = [];
        lambda_mat = [];
        angle_mat = [];

        syms ks
        tic
        for k = 1:size(Tree,1)

            if Tree(k,2)*Tree(k,3)~=0  %����Ҫ�ų�������
               PD = double(Tree(k,4));
               DD1 = Tree(Tree(k,2),4); % ��Ϊ�������Ҫ��ȡ��һ���Ĺܾ���Ϣ��K��lambda����Ϣû�д�����
               DD2 = Tree(Tree(k,3),4);

               k_mat(end+1) = double(vpasolve(PD^ks-DD1^ks-DD2^ks==0,ks));
               lambda_mat(end+1) = min(DD1,DD2)/max(DD1,DD2);

               ax = Coord{Tree(k,2),1};
               ay = Coord{Tree(k,2),2};

               bx = Coord{Tree(k,3),1};
               by = Coord{Tree(k,3),2};

               seg1 = [ax(2)-ax(1),ay(2)-ay(1)];
               seg2 = [bx(2)-bx(1),by(2)-by(1)];
               angle_mat(end+1) = acos((seg1(1)*seg2(1)+seg1(2)*seg2(2))/((seg1(1)^2+seg1(2)^2)*(seg2(1)^2+seg2(2)^2))^0.5)/pi*180;

            end

        end
        toc
        k_cv(end+1) = std(k_mat)/mean(k_mat);
        lambda_cv(end+1) = std(lambda_mat)/mean(lambda_mat);
        angle_cv(end+1) = std(angle_mat)/mean(angle_mat);

        k_mean(end+1)=mean(k_mat);
        k_mean(end+1)=std(k_mat);
        lambda_mean(end+1)=mean(lambda_mat);
        lambda_mean(end+1)=std(lambda_mat);
        angle_mean(end+1)=mean(angle_mat);
        angle_mean(end+1)=std(angle_mat);


    end

end

CV_mat(1,:) = k_cv;
CV_mat(2,:) = lambda_cv;
CV_mat(3,:) = angle_cv;
save(sprintf('../Data/Temp/CV_%s2.mat',modeltype),'CV_mat') 

