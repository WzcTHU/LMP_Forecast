load('LGBMres.mat');
load('MLPres.mat');
load('RFres.mat');
load('SVMres.mat');
load('XGBres.mat');
load('RNNres.mat');
V = [LGBM_res, MLP_res, RF_res, SVM_res, XGB_res, RNN_res]';
H = V*V';
W = sdpvar(6, 1);
C = [sum(W) == 1
     W >= 0];
options = sdpsettings('solver', 'cplex', 'showprogress',1);
z = W' * H * W;
result = optimize(C, z, options);
if result.problem == 0
    W_value = value(W)
    value(z)
else
    disp('wrong');
end

save('models_W.mat', 'W_value');