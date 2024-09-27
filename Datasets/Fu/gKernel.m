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
        % ��ⷶ��
        sl(i)=norm(inter_lncdis(i,:))^2;
    end
    % ���㹫ʽnl
    gamal=nl/sum(sl')*1;
    for i=1:nl
        for j=1:nl
            % ���㹫ʽLGS
            pkl(i,j)=exp(-gamal*(norm(inter_lncdis(i,:)-inter_lncdis(j,:)))^2);
        end
    end        
    for i=1:nd
        % ���㷶��
        sd(i)=norm(inter_lncdis(:,i))^2;
    end
    % ���㹫ʽnd
    gamad=nd/sum(sd')*1; 
    for i=1:nd
        for j=1:nd
            % ���㹫ʽDGS
            pkd(i,j)=exp(-gamad*(norm(inter_lncdis(:,i)-inter_lncdis(:,j)))^2);
        end
    end 
    % ��LGS����Ϊresult_lnc��DGS����Ϊresult_dis
    result_lnc=pkl;
    result_dis=pkd;
end
   

