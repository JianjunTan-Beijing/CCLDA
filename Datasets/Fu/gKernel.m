function [result_lnc,result_dis]=gKernel(nl,nd,inter_lncdis)
%gKernel compute Gaussian interaction profile kernel
%   Usage:  [result_lnc,result_dis]=gKernel(nl,nd,inter_lncdis)
%	Inputs:
%			nl: the number of lncRNAs
%			nd:	the number of diseases
%			inter_lncdis: an nl*nd association matrix between lncRNAs and diseases
%
%	Outputs:
%			result_lnc: Gaussian interaction profile kernel of lncRNAs
%			result_dis: Gaussian interaction profile kernel of diseases


    for i=1:nl
        % 求解范数
        sl(i)=norm(inter_lncdis(i,:))^2;
    end
    % 计算公式nl
    gamal=nl/sum(sl')*1;
    for i=1:nl
        for j=1:nl
            % 计算公式LGS
            pkl(i,j)=exp(-gamal*(norm(inter_lncdis(i,:)-inter_lncdis(j,:)))^2);
        end
    end        
    for i=1:nd
        % 计算范数
        sd(i)=norm(inter_lncdis(:,i))^2;
    end
    % 计算公式nd
    gamad=nd/sum(sd')*1; 
    for i=1:nd
        for j=1:nd
            % 计算公式DGS
            pkd(i,j)=exp(-gamad*(norm(inter_lncdis(:,i)-inter_lncdis(:,j)))^2);
        end
    end 
    % 将LGS保存为result_lnc；DGS保存为result_dis
    result_lnc=pkl;
    result_dis=pkd;
end
   

