%% Orthogonal Matching Pursuit for CS
%% 
%%
%% Parameters
N = 60;
M = 30;
P = 40;
% density of non-zero components
rho = 0.5;
k = N*rho;

THETA = 0.001;
SAMPLE = 1000;

L2 = zeros(P,1);

str = sprintf('CS_N%d_M%d_P%d_rho%.2f.dat', N, M, P, rho);
fid = fopen(str, 'W');

for t = 1: SAMPLE

    X0 = randn(N,P);
    X0 = (rand(N,P)<rho).*X0;
    D = randn(M,N);
    Y = D*X0;
    X = zeros(N,P);
    
    e = zeros(N,1);  %% contribution of each element on error

    for l = 1: P
        x = X(:,l);
        r = Y(:,l)-D*x;
        S = [];      %% indices of the non-zero elements
        while size(S,1) <= k && norm(r) > THETA
            for i = 1: N
                e(i) = -norm(D(:,i)'*r)/norm(D(:,i));
            end
            [C,I] = min(e);
            S = [S;I];
            S = sort(S);
            DS = D(:,S(:));
            XS = (DS'*DS)\(DS'*Y(:,l));
            x(S(:)) = XS(:);
            r = Y(:,l)-D*x;
        end
        X(:,l) = x;
        L2(l) = norm(X(:,l)-X0(:,l))/sqrt(N);
    end
    fprintf(fid,'%f\n', sum(L2)/P);
end
fclose(fid);