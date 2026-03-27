function [label,all_time]=AGC(X,p,alpha,lambda,rho)

%   X       - Dataset matrix
%   p       - Number of clusters
%   alpha   - Angle between planes
%   lambda  - Regularization coefficient
%   rho     - Regularization coefficient
%   label   - Clustering result labels
%   all_time- Total algorithm running time (s)

tolerance = 1e-4;
n = size(X, 1);
d = size(X, 2);

tic
[theta]=initialize_theta(n);
[W]=initialize_W(n, p);
[WW]=initialize_WW(p,d);

for iteration=1:500
    P=update_P(theta,p,W,X);
    X=X';
    P=P';
    [QA]=update_QA(X,theta,p,P,n,d,W);
    G = cos(alpha) * ones(d);  
    G(1:d+1:end) = 1;
    G_half = sqrtm(G);
    Z= reshape( ...
        bsxfun(@times, eye(d), reshape(WW,1,d,p)), ...
        d, d, p);
    M=pagemtimes(pagemtimes(G_half, Z),G_half);
    [QB,~] = pagesvd(M);
    QB = pagetranspose(QB);
    G_half=repmat(G_half, [1 1 p]);
    QQ=pagemtimes(QB,G_half);
    V= pagemtimes(QA, QQ);  
    dist1=compute_dist1(WW,d,p,X,n,V,P);
    W=update_W(theta,p,n,dist1);
    [C]=compute_dist2(W,theta,d,p,X,n,V,P);
    WW=update_WW(lambda,d,C);
    dist1=compute_dist1(WW,d,p,X,n,V,P);
    for iter = 1:1
         m=theta+1;
         E = repmat(m, p, 1);
         W3=(W.^ E).*log(W+0.00001);
         grad=sum((dist1) .* W3)+rho*sign(theta);
         h=sum((dist1) .* (W3.*log(W+0.00001)));
         h=h.^(-1); 
         theta=max(theta-h.*grad,0);
         objective=compute_objective(WW,theta,p,W,dist1,rho,lambda);
    end
    if iteration > 1
            diff = abs(objective - objective_prev2);
            fprintf('Iteration %d: current objective = %.6f, previous objective = %.6f, diff = %.6f\n', ...
            iteration, objective, objective_prev2, diff);
            if diff < tolerance
            break;
            end
     end
     objective_prev2 = objective;  % 更新前一次值
     X=X';
end
     W=W';
     [~,label]=max(W,[],2);
     all_time=toc;
     disp('end')
end

function [theta] = initialize_theta(n)
          theta = ones(1, n);
end

function [W] = initialize_W(n, p)
          W = rand(n, p);
          row_sums = sum(W, 2);
          W = W./row_sums;
          W=W';
end

function [WW]=initialize_WW(p,d)
          WW = rand(p,d);
          row_sums = sum(WW, 2);
          WW = WW./row_sums;
          WW=WW';
end

function [P]=update_P(theta,p,W,X)
          m=theta+1;
          E = repmat(m, p, 1);
          W=W.^ E;
          W(all(W==0,2),:)=eps;
          P = W*X./(sum(W,2)*ones(1,size(X,2)));
end

function [QA]=update_QA(X,theta,p,P,n,d,W)
          m=theta+1;  
          E = repmat(m, p, 1);
          W=W.^ E;
          X_1 = repmat(X, 1, p);
          P_1 = kron(P, ones(1, n));
          Y = reshape((X_1-P_1), [], 1, n*p) .* reshape((X_1-P_1), 1, [], n*p);
          W0=reshape(W.', 1, []);
          W_1 = reshape(W0, 1, 1, []);   % 1×1×2
          Y_1 = W_1 .* Y;
          Y_2=squeeze(sum(reshape(Y_1, d, d, n, p), 3));   
          [QA, ~] = pagesvd(Y_2);
          QA=QA(:, end:-1:1, :); 
end

function [dist1]=compute_dist1(WW,d,p,X,n,V,P)
          X_1 = repmat(X, 1, p);
          P_1 = kron(P, ones(1, n));
          PX=X_1-P_1;
          PX=PX(:);
          PX1 = reshape(PX, d, p*n).';      
          PX1 = repelem(PX1, d, 1);        
          PX1 = reshape(PX1.', 1, []).';    
          V1=reshape(V, 1, []);
          V2 = reshape(V1, d*d, p);  
          V3 = repelem(V2, 1, n);        
          V4 = reshape(V3, 1, []);       
          total_segments = d * p * n;  
          segment_size = d;
          A_segments = reshape(V4, segment_size, total_segments); 
          B_segments = reshape(PX1, segment_size, total_segments); 
          dots = sum(A_segments .* B_segments, 1);
          dots_grouped = reshape(dots, d, p*n); 
          dots_grouped1=dots_grouped.^2;
          A_3d = reshape(dots_grouped1, d, n, p);
          B_3d = reshape(WW, d, 1, p);
          C_3d = B_3d .* A_3d;
          CC = reshape(C_3d, d, p*n);
          CC = sum(CC, 1);        
          dist1=reshape(CC, n, p).';
end

function [W]=update_W(theta,p,n,dist)
          i1 = find(theta == 0);
          if isempty(i1)
          Q1 = zeros(p, n);  
          else
          [~, j] = min(dist(:, i1));
          Q1 = zeros(p,n);        
          linear_idx = (i1 - 1) * p + j;
          Q1(linear_idx) = 1;       
          end
          i2 = find(theta ~= 0);
          if isempty(i2)
          Q3 = zeros(p, n); 
          else
          dist_i2 = dist(:, i2)+0.00001;
          n2= length(i2);
          m=theta+1;
          E = repmat(m, p, 1);
          E_i2=E(:, i2);
          tmp1 = repmat((dist_i2(:)).', p, 1);
          tmp2 = kron(dist_i2, ones(1, p));
          tmp3 = 1./(E_i2-1);
          tmp3_1 = kron(tmp3(:).', ones(p, 1));
          tmp3_2 = reshape(((sum((tmp1 ./ (tmp2)).^tmp3_1, 1)).^(-1)).', p, n2);
          Q3 = zeros(p,n);
          Q3(:, i2) = tmp3_2(:, 1:length(i2));
          end
          W=Q3+Q1;
end

function [objective]=compute_objective(WW,theta,p,W,dist1,rho,lambda)
          m=theta+1;
          E = repmat(m, p, 1);
          W4=W.^ E;
          objective=sum(sum((dist1) .* W4))+rho*sum(abs(theta))+lambda*sum(sum(WW.*log(WW+0.00001)));
end

function [C]=compute_dist2(W,theta,d,p,X,n,V,P)
          X_1 = repmat(X, 1, p);
          P_1 = kron(P, ones(1, n));
          PX=X_1-P_1;
          PX=PX(:);
          PX1=reshape(PX, d, p*n).';      
          PX1=repelem(PX1, d, 1);       
          PX1=reshape(PX1.', 1, []).';      
          V1=reshape(V, 1, []);
          V2=reshape(V1, d*d, p);  
          V3=repelem(V2, 1, n);       
          V4=reshape(V3, 1, []);      
          total_segments = d * p * n;  
          segment_size = d;
          A_segments = reshape(V4, segment_size, total_segments);
          B_segments = reshape(PX1, segment_size, total_segments); 
          dots = sum(A_segments .* B_segments, 1);
          dots_grouped = reshape(dots, d, p*n);  
          dots_grouped1=dots_grouped.^2;
          m=theta+1;
          E = repmat(m, p, 1);
          W=W.^E;
          W=W';
          W=W(:).';
          C = dots_grouped1 .* repmat(W, d, 1);
          C = squeeze(sum(reshape(C, d, n, p), 2));
end

function [WW]=update_WW(lambda,d,C)
          tmp=exp(C / (-lambda));
          tmp(isnan(tmp)) = 0;
          WW=tmp./(ones(d,1)*(eps+sum(tmp)));
end

