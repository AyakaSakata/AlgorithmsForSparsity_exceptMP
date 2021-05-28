%% Method of optimal direction for Dictionary Learning
%% 
%%
clear variables;
close all;
%% Parameters
N = 60;
M = 30;
% number of samples for training 
P = 60;
rho = 0.1;
theta = 0.1;
k = N*theta;
D_sigma = 0.1;

sqN = sqrt(N);
sqM = sqrt(M);

THETA_X = 1e-5;
THETA_Y = 1e-4;
SAMPLE = 1;
count_max = 1000;

L2 = zeros(P, 1);
dif = zeros(N, N);
Dp = zeros(M, N);
J = zeros(N, 1);

str = sprintf('DL_N%d_M%d_P%d_rho%.2f_theta%.2f_sigma%.2f.dat', N, M, P, rho, theta, D_sigma);
%str = sprintf('DL_N%d_M%d_P%d_rho%.2f_random.dat', N, M, P, rho);
fid = fopen(str, 'W');

for t = 1: SAMPLE
    
    count = 0;
    
    % true dictionary
    X0 = zeros(N,P);
    X0(1:k,:) = randn(k,P);
    for j = 1: P
        X0(:,j) = X0(randperm(N),j);
    end    
    
    % true sample
    D0 = randn(M,N);
    for i = 1: N
        D0(:,i) = sqM*D0(:,i)/norm(D0(:,i));
    end
    D0 = D0/sqN;  % sqN is the coefficient of Y = D0*X0/sqN, and D0/sqN is defined as D0.
    Y = D0*X0;
    
    % initial condition of D
    D = randn(M,N);
    for i = 1: N
        D(:,i) = sqM*D(:,i)/norm(D(:,i));
    end
    D = D/sqN;
    
    X = zeros(N,P);
    e = zeros(N,1);  %% contribution of each element on error

%    while trace((D*X/sqN-Y)'*(D*X/sqN-Y))/(M*P) > THETA_Y && count < count_max
    while count < count_max
        count = count + 1;
       %% X update: Orthogonal Matching Pursuit
        X = zeros(N,P);
        for l = 1: P
            x = X(:,l);
            r = Y(:,l)-D*x;
            S = [];      %% indices of the non-zero elements
            while size(S,1) < k && r'*r/N > THETA_X
                for i = 1: N
                    e(i) = -norm(D(:,i)'*r)/norm(D(:,i));
                end
                [C,I] = min(e);
                if abs(C) < 1e-6
                    break;
                else
                    S = [S;I];
                    S = sort(S);
                    DS = D(:,S(:));
                    XS = (DS'*DS)\(DS'*Y(:,l));
                    %XS = pinv(DS)*Y(:,l);
                    x(S(:)) = XS(:);
                    r = Y(:,l)-D*x;
                end
            end
            X(:,l) = x;
            %L2(l) = norm(X(:,l)-X0(:,l))/sqrt(N);
        end
      %% D update: Method of Optimal Direction
       gamma = rand(1,1);
       D = gamma*D+(1-gamma)*Y*pinv(X);
       %D = gamma*D+(1-gamma)*Y/inv(X);
       
       %D = normD*D/sqrt(trace(D'*D));
       for i = 1: N
           D(:,i) = sqM*D(:,i)/norm(D(:,i));
       end
       D = D/sqN;
       
       % Permutation & parity
        [permindx,parity] = perm(D0,D,N);
        % Maximizing overlap by the permutation 
        D_perm=D(:,permindx)*diag(parity);
        X_perm = zeros(N,P);
        for i = 1: N
            X_perm(i,:) = x(permindx(i),:)*parity(i);
        end
        
        % similarity between true signal/dictionary and the estimated one
        similarity_D=zeros(N,1);
        similarity_X=zeros(N,1);
        power_X = zeros(N,1);
        for i=1:N
            similarity_D(i)=D0(:,i)'*D_perm(:,i)*N;
            similarity_X(i)=x0(i,:)*X_perm(i,:)';
            power_X(i) = X_perm(i,:)*X_perm(i,:)';
        end
        fprintf(fid_temp, '%d %f %f\n', count, sum(similarity_D)/(M*N), sum(similarity_X)/(N*P));
        
    end
    
end
fclose(fid);
