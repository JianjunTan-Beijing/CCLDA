% �����������������
clc,clear all
% ����nd.xlsx������ֵ���ı�������ȡ���ֱ�����Nd��txt
[Nd,txt,~]=xlsread('diseases_name.xlsx');
% ֻ�����ı�
ND_txt=txt;
% ndΪ���ı��ĳ���
nd=length(ND_txt);
% ͬ���nl.xlsx���в���
[Nl,txt,~]=xlsread('lncRNA_name.xlsx');
% nl.xlsx�е�һ���Ǳ�ǩ��ȥ��
NL_txt=txt(2:end,1);
% ȡ����
nl=length(NL_txt);
% ��ȡ01����
[Inter_lncdis,txt,~]=xlsread('lnc_di A.xlsx');
% ȥ����һ��
Inter_lncdis(1,:)=[];
% ȥ����һ��
Inter_lncdis(:,1)=[];
% ���ú���
[result_lnc,result_dis]=gKernel(nl,nd,Inter_lncdis);
% �ֱ�ѽ��������LGS.xlsx��DGS.xlsx��
xlswrite('LGS.xlsx',result_lnc);
xlswrite('DGS.xlsx',result_dis);