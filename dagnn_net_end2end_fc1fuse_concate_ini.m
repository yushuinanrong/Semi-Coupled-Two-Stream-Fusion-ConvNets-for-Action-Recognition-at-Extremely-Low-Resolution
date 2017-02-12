function nn = dagnn_net_end2end_fc1fuse_concate_ini(varargin)
% Author: Jiawei Chen
% Two stream ConvNet fused after first fully-connected layer using concatenation fusion 

L = 5;
% init the object:
nn = dagnn.DagNN();
%------------Spatial Block1--------------------------------
c_cnv1 = dagnn.Conv('size',[5 5 3 32],'pad',2,'stride',1,'hasBias',true);
nn.addLayer('c_cnv1', c_cnv1, {'input_1'}, {'xc1_1'},{'c1','b1'}); % output dim = 30-6 = 24
nn.addLayer('c_bn1', dagnn.BatchNorm('numChannels',32), {'xc1_1'}, {'xc2_1'},{'bn1f', 'bn1b', 'bn1m'});

nn.addLayer('c_re1', dagnn.ReLU(), {'xc2_1'}, {'xc3_1'});
c_mp1 = dagnn.Pooling('method', 'max', 'poolSize', [3 3],'pad', 1, 'stride', 2);
nn.addLayer('c_mp1', c_mp1, {'xc3_1'}, {'xc4_1'}); % output dim = 24/2 = 12
nn.addLayer('dropout1', dagnn.DropOut('rate', 0.5), {'xc4_1'}, {'dp1'}); 

%------------Spatial Block2--------------------------------
c_cnv2 = dagnn.Conv('size',[5 5 32 64],'pad',2,'stride',1,'hasBias',true);
nn.addLayer('c_cnv2', c_cnv2, {'dp1'}, {'xc5_1'},{'c2','b2'}); % output dim = 30-6 = 24
nn.addLayer('c_bn2', dagnn.BatchNorm('numChannels',64), {'xc5_1'}, {'xc6_1'},{'bn2f', 'bn2b', 'bn2m'});
nn.addLayer('c_re2', dagnn.ReLU(), {'xc6_1'}, {'xc7_1'});
c_mp2 = dagnn.Pooling('method', 'avg', 'poolSize', [3 3],'pad', 1, 'stride', 2);
nn.addLayer('c_mp2', c_mp2, {'xc7_1'}, {'xc8_1'}); % output dim = 24/2 = 12
nn.addLayer('dropout2', dagnn.DropOut('rate', 0.5), {'xc8_1'}, {'dp2'}); 

%------------Spatial Block3--------------------------------
c_cnv3 = dagnn.Conv('size',[5 5 64 128],'pad',2,'stride',1,'hasBias',true);
nn.addLayer('c_cnv3', c_cnv3, {'dp2'}, {'xc9_1'},{'c3','b3'}); % output dim = 30-6 = 24
nn.addLayer('c_bn3', dagnn.BatchNorm('numChannels',128), {'xc9_1'}, {'xc10_1'},{'bn3f', 'bn3b', 'bn3m'});
nn.addLayer('c_re3', dagnn.ReLU(), {'xc10_1'}, {'xc11_1'});
c_mp3 = dagnn.Pooling('method', 'avg', 'poolSize', [3 3],'pad', 1, 'stride', 2);
nn.addLayer('c_mp3', c_mp3, {'xc11_1'}, {'xc12_1'}); % output dim = 24/2 = 12

%------------ Spatial fully connected layer1---------------
c_fc1 = dagnn.Conv('size',[4 4 128 128],'pad',0,'stride',1,'hasBias',true);
nn.addLayer('c_fc1', c_fc1, {'xc12_1'}, {'xc13_1'},{'c4','b4'});
nn.addLayer('c_re4', dagnn.ReLU(), {'xc13_1'}, {'xc14_1'});

%------------Temporal Block1--------------------------------
t_cnv1 = dagnn.Conv('size',[5 5 3*(2*L+1) 32],'pad',2,'stride',1,'hasBias',true);
nn.addLayer('t_cnv1', t_cnv1, {'input_2'}, {'xc1_2'},{'c5','b5'}); % output dim = 30-6 = 24
% nn.addLayer('t_bn1', dagnn.BatchNorm('numChannels',32), {'xc1_2'}, {'xc2_2'},{'bn4f', 'bn4b', 'bn4m'});
nn.addLayer('t_re1', dagnn.ReLU(), {'xc1_2'}, {'xc3_2'});
t_mp1 = dagnn.Pooling('method', 'max', 'poolSize', [3 3],'pad', 1, 'stride', 2);
nn.addLayer('t_mp1', t_mp1, {'xc3_2'}, {'xc4_2'}); % output dim = 24/2 = 12
nn.addLayer('dropout3', dagnn.DropOut('rate', 0.2), {'xc4_2'}, {'dp3'}); 

%------------Temporal Block2--------------------------------
t_cnv2 = dagnn.Conv('size',[5 5 32 64],'pad',2,'stride',1,'hasBias',true);
nn.addLayer('t_cnv2', t_cnv2, {'dp3'}, {'xc5_2'},{'c6','b6'}); % output dim = 30-6 = 24
nn.addLayer('t_bn2', dagnn.BatchNorm('numChannels',64), {'xc5_2'}, {'xc6_2'},{'bn5f', 'bn5b', 'bn5m'});
nn.addLayer('t_re2', dagnn.ReLU(), {'xc6_2'}, {'xc7_2'});
t_mp2 = dagnn.Pooling('method', 'avg', 'poolSize', [3 3],'pad', 1, 'stride', 2);
nn.addLayer('t_mp2', t_mp2, {'xc7_2'}, {'xc8_2'}); % output dim = 24/2 = 12
nn.addLayer('dropout4', dagnn.DropOut('rate', 0.2), {'xc8_2'}, {'dp4'}); 

%------------Temporal Block3--------------------------------
t_cnv3 = dagnn.Conv('size',[5 5 64 64],'pad',2,'stride',1,'hasBias',true);
nn.addLayer('t_cnv3', t_cnv3, {'dp4'}, {'xc9_2'},{'c7','b7'}); % output dim = 30-6 = 24
nn.addLayer('t_bn3', dagnn.BatchNorm('numChannels',64), {'xc9_2'}, {'xc10_2'},{'bn6f', 'bn6b', 'bn6m'});
nn.addLayer('t_re3', dagnn.ReLU(), {'xc10_2'}, {'xc11_2'});
t_mp3 = dagnn.Pooling('method', 'avg', 'poolSize', [3 3],'pad', 1, 'stride', 2);
nn.addLayer('t_mp3', t_mp3, {'xc11_2'}, {'xc12_2'}); % output dim = 24/2 = 12

%------------ Temporal fully connected layer1---------------
t_fc1 = dagnn.Conv('size',[4 4 64 128],'pad',0,'stride',1,'hasBias',true);
nn.addLayer('t_fc1', t_fc1, {'xc12_2'}, {'xc13_2'},{'fc1','fc11'});
nn.addLayer('t_re4', dagnn.ReLU(), {'xc13_2'}, {'xc14_2'});

%------------Fusion Block----------------------------------------
%--do cancate on the first fully connected layers
nn.addLayer('concate', dagnn.Concat('dim',3), {'xc14_1', 'xc14_2'}, {'cv_concat'}); 
% f_fcc = dagnn.Conv('size', [1,1,256,128],'pad',0,'stride',1,'hasBias',true);
% nn.addLayer('f_fcc',f_fcc, {'cv_concat'}, {'conv_fc1'},{'s','t'}); 
nn.addLayer('f_re1', dagnn.ReLU(), {'conv_fc1'}, {'fc1_relu'});
nn.addLayer('dropout5', dagnn.DropOut('rate', 0.5), {'fc1_relu'}, {'dp5'}); 

%------------ fully connect 2---------------------------------------------
f_fc2 = dagnn.Conv('size',[1 1 256 12],'pad',0,'stride',1,'hasBias',true);
nn.addLayer('f_fc2', f_fc2, {'dp5'}, {'pred'},{'d','prob'});

%--------------------softmax loss-----------------------------
nn.addLayer('loss',  dagnn.Loss('loss', 'softmaxlog'), {'pred','label'}, 'objective');
nn.addLayer('error', dagnn.Loss('loss', 'classerror'), {'pred','label'}, 'error');

% *****************************************************************************
% initialize the weights:
nn.initParams();
nn.meta.trainOpts.learningRate = [0.01*ones(1,10) 0.005*ones(1,10) 0.0005*ones(1,5)];
nn.meta.trainOpts.weightDecay  = 0.0005 ;
nn.meta.trainOpts.batchSize    = 256 ;
nn.meta.trainOpts.numEpochs    = numel(nn.meta.trainOpts.learningRate);