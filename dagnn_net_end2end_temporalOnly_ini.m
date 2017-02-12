function nn = dagnn_net_end2end_temporalOnly_ini(varargin)

% init the object:
nn = dagnn.DagNN();
% %------------Temporal Block1--------------------------------
t_cnv1 = dagnn.Conv('size',[5 5 3 16],'pad',2,'stride',1,'hasBias',true);
nn.addLayer('t_cnv1', t_cnv1, {'input_2'}, {'xc1_2'},{'c5','b5'}); % output dim = 30-6 = 24
nn.addLayer('t_bn1', dagnn.BatchNorm('numChannels',16), {'xc1_2'}, {'xc2_2'},{'bn4f', 'bn4b', 'bn4m'});
nn.addLayer('t_re1', dagnn.ReLU(), {'xc2_2'}, {'xc3_2'});
t_mp1 = dagnn.Pooling('method', 'max', 'poolSize', [3 3],'pad', 1, 'stride', 2);
nn.addLayer('t_mp1', t_mp1, {'xc3_2'}, {'xc4_2'}); % output dim = 24/2 = 12
nn.addLayer('dropout3', dagnn.DropOut('rate', 0.2), {'xc4_2'}, {'dp3'}); 

%------------Temporal Block2--------------------------------
t_cnv2 = dagnn.Conv('size',[5 5 16 32],'pad',2,'stride',1,'hasBias',true);
nn.addLayer('t_cnv2', t_cnv2, {'dp3'}, {'xc5_2'},{'c6','b6'}); % output dim = 30-6 = 24
nn.addLayer('t_bn2', dagnn.BatchNorm('numChannels',32), {'xc5_2'}, {'xc6_2'},{'bn5f', 'bn5b', 'bn5m'});
nn.addLayer('t_re2', dagnn.ReLU(), {'xc6_2'}, {'xc7_2'});
t_mp2 = dagnn.Pooling('method', 'avg', 'poolSize', [3 3],'pad', 1, 'stride', 2);
nn.addLayer('t_mp2', t_mp2, {'xc7_2'}, {'xc8_2'}); % output dim = 24/2 = 12
nn.addLayer('dropout4', dagnn.DropOut('rate', 0.2), {'xc8_2'}, {'dp4'}); 

%------------Temporal Block3--------------------------------
t_cnv3 = dagnn.Conv('size',[5 5 32 64],'pad',2,'stride',1,'hasBias',true);
nn.addLayer('t_cnv3', t_cnv3, {'dp4'}, {'xc9_2'},{'c7','b7'}); % output dim = 30-6 = 24
nn.addLayer('t_bn3', dagnn.BatchNorm('numChannels',64), {'xc9_2'}, {'xc10_2'},{'bn6f', 'bn6b', 'bn6m'});
nn.addLayer('t_re3', dagnn.ReLU(), {'xc10_2'}, {'xc11_2'});
t_mp3 = dagnn.Pooling('method', 'avg', 'poolSize', [3 3],'pad', 1, 'stride', 2);
nn.addLayer('t_mp3', t_mp3, {'xc11_2'}, {'xc12_2'}); % output dim = 24/2 = 12

%------------ Temporal fully connected layer1---------------
t_fc1 = dagnn.Conv('size',[4 4 64 64],'pad',0,'stride',1,'hasBias',true);
nn.addLayer('t_fc1', t_fc1, {'xc12_2'}, {'xc13_2'},{'fc1','fc11'});
nn.addLayer('t_re4', dagnn.ReLU(), {'xc13_2'}, {'xc14_2'});
nn.addLayer('dropout5', dagnn.DropOut('rate', 0.5), {'xc14_2'}, {'dp5'}); 

%------------ fully connect 2---------------------------------------------
f_fc2 = dagnn.Conv('size',[1 1 256 51],'pad',0,'stride',1,'hasBias',true);
nn.addLayer('f_fc2', f_fc2, {'dp5'}, {'pred'},{'d','prob'});

%--------------------softmax loss-----------------------------
nn.addLayer('loss',  dagnn.Loss('loss', 'softmaxlog'), {'pred','label'}, 'objective');
nn.addLayer('error', dagnn.Loss('loss', 'classerror'), {'pred','label'}, 'error');

% *****************************************************************************
% initialize the weights:
nn.initParams();
nn.meta.trainOpts.learningRate = [0.05*ones(1,15) 0.005*ones(1,15) 0.0005*ones(1,10),0.00005*ones(1,10)];
nn.meta.trainOpts.weightDecay  = 0.0005 ;
nn.meta.trainOpts.batchSize    = 256;
nn.meta.trainOpts.numEpochs    = numel(nn.meta.trainOpts.learningRate);