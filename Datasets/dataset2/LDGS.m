% 清楚工作区和命令行
clc,clear all
% 由于nd.xlsx中有数值和文本，均读取，分别存放在Nd和txt
[Nd,txt,~]=xlsread('diseases_name.xlsx');
% 只保留文本
ND_txt=txt;
% nd为该文本的长度
nd=length(ND_txt);
% 同理对nl.xlsx进行操作
[Nl,txt,~]=xlsread('lncRNA_name.xlsx');
% nl.xlsx中第一行是标签，去除
NL_txt=txt(2:end,1);
% 取长度
nl=length(NL_txt);
% 读取01矩阵
[Inter_lncdis,txt,~]=xlsread('lnc_di A.xlsx');
% 去除第一行
Inter_lncdis(1,:)=[];
% 去除第一列
Inter_lncdis(:,1)=[];
% 调用函数
[result_lnc,result_dis]=gKernel(nl,nd,Inter_lncdis);
% 分别把结果保存在LGS.xlsx和DGS.xlsx中
xlswrite('LGS.xlsx',result_lnc);
xlswrite('DGS.xlsx',result_dis);