clear;
clc;
% 计算分叉的分叉指数和不对称比例
%% =====================


Net = ["Men_389","Men_546","Men_913"];
NetType = ["Bifurcation","Convergence"];


for net = 1:length(Net)
    for nt = 1:length(NetType)
        tic
        Data = load(sprintf('../Data/Normalized Data/Diam_%s_%s.mat',Net(net),NetType(nt)));
        Diam = Data.Diam;
               
        Norm_BE = zeros(1,size(Diam,1));
        Asy_ratio = zeros(1,size(Diam,1));
        min_tag = zeros(1,size(Diam,1));
        
        syms k
        parfor i = 1:size(Diam,1)
            PD = Diam(i,1);
            DD1 = Diam(i,2);
            DD2 = Diam(i,3);
            
            if DD1>=PD
                DD1 = PD*0.81;
            end
            if DD2>=PD
                DD2 = PD*0.79;
            end           
            
            Bif_Exponent = double(vpasolve(PD^k-DD1^k-DD2^k==0,k));
            
            if Bif_Exponent<1.5
                Norm_BE(i) = 0;
            elseif Bif_Exponent>6
                Norm_BE(i) = 1;
            else
                Norm_BE(i) = (Bif_Exponent-1.5)/4.5;
            end          
            
            Asy_ratio(i) = min(DD1,DD2)/max(DD1,DD2);
            
            if min(DD1,DD2)==DD1
                min_tag(i) = 1;
            else
                min_tag(i) = 2;
            end
        end      
        
        Diam(:,4) = Norm_BE;
        Diam(:,5) = Asy_ratio;
        Diam(:,6) = min_tag;
        save(sprintf('../Data/Normalized Data/k&lambda_%s_%s.mat',Net(net),NetType(nt)),'Diam','-v7.3')
        toc
    end
end

