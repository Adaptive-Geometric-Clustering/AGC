% function results = ClusteringMeasure(true_labels,cluster_labels,beta)
% results = [NMI,ACC, RandIndx,Purity, Fbeta, Precision, Recall, AdjRandIndx]
% This code calculates the mainly clustering mearsures (accuarcy, NMI, Rand Index, Purity, Fvalue) between two cluster partitions
% Date 29 March 2017(Hong Tao)
function results = ClusteringMeasure(true_labels,cluster_labels,beta)
if ~exist('beta','var')
beta = 1;
end
true_labels = true_labels(:);
cluster_labels = cluster_labels(:);
[purity1, cls] = purity(true_labels,cluster_labels);
[m, n] = size(cls);
no_objects = sum(sum(cls));
i_dot = sum(cls,2);
j_dot = sum(cls,1);
tr = zeros(m,n);
for i = 1:m
for j = 1:n
if cls(i,j)<2
tr (i,j) = 0;
else
tr (i,j) = nchoosek(cls(i,j),2);
end
end
end
term1 = sum(sum(tr));
for in = 1:length(i_dot)
if i_dot(in)<2
c_i(in) = 0;
else
c_i(in) = nchoosek(i_dot(in),2); 
end
end
for jn = 1:length(j_dot)
if j_dot(jn)<2
c_j(jn) = 0;
else
c_j(jn) = nchoosek(j_dot(jn),2); 
end
end
TP = sum(sum(tr));
FP = sum(c_i) - sum(sum(tr));
FN = sum(c_j) - sum(sum(tr));
Precision = TP/(TP+FP);
Recall = TP/(TP+FN);
P = Precision;
R = Recall;
Fbeta = (beta^2 + 1)*P*R/(beta^2*P + R);
ARI_t = term1 - (sum(c_i)*sum(c_j))/nchoosek(no_objects,2);
ARI_d = 0.5*(sum(c_i) + sum(c_j)) - (sum(c_i)*sum(c_j))/nchoosek(no_objects,2);
AdjRandIndx = ARI_t/ARI_d;
% if numel(unique(true_labels)) == numel(unique(cluster_labels))
% ACC = accuracy(true_labels,cluster_labels);
% else
% ACC = 0;
 res = bestMap(true_labels,cluster_labels); %gnd 原始标签 index :n*1 得到标签 % gnd:n*1
 ACC = length(find(true_labels == res))/length(true_labels);
% end
[NMI] = nmi( cluster_labels, true_labels);
%% to calculate the chi coefficient
results=[ACC,NMI,Fbeta,AdjRandIndx,purity1];
% nSmp = length(true_labels);
% Cluster1 = length(unique(true_labels));
% Ytrue = zeros(nSmp,Cluster1);
% for i = 1:Cluster1
% Ytrue(true_labels == i,i) = 1;
% end
% 
% ncluster2 = length(unique(cluster_labels));
% Y = zeros(nSmp,ncluster2);
% for i = 1:ncluster2
% Y(cluster_labels == i,i) = 1;
% end
% JI = JaccardIndex( Ytrue, Y );
% results = [NMI,MI, ACC, RandIndx,Purity, Fbeta, Precis
