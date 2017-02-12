%% Alternating Descent Two stream connected IXMAS
%Author: Jiawei Chen

clc;close all;
imdb_low = load('your\data\path\eLRdata');
imdb_high= load('your\data\path\HRdata');



for Testpp  = 1:10 % change this according to your dataset
    
    Tempstrach       = true;
    if Tempstrach == true % IXMAS dataset
        tempLen = 100*ones(1800,1);
    elseif Tempstrach == false
        tempLen = imdb.images.tempLen;
    end
    
    Ind     = cumsum(tempLen); % video Ind
    Ind2    = reshape(Ind,180,10);
    Ind2    = Ind2(180,:);
    set     = ones(1,length(imdb_low.images.set));
    if Testpp == 1
        set(1:Ind2(1)) = 2;
    else
        set(Ind2(Testpp-1)+1:Ind2(Testpp))=2;
    end
    
    imdb_low.images.set    = set;
    imdb_high.images.set   = set;
    opts.networkType = 'dagnn';
    opts.modelType   = 'end2end_lascov_fusion_sum' ;
    %% Part 3.3: learn the model
    % Train
    switch opts.modelType
        
        case 'end2end_lascov_fusion_sum'
            net = dagnn_net_end2end_fuse_lastcv_sum_opticalstack_ini();
            
    end
    opts1.expDir  = fullfile('your\data\path') ;
    trainOpts1.learningRate   = [1*1E-3*ones(1,10) 1E-4*ones(1, 10) 1E-5*ones(1, 5)] ;
    opts1.train = struct() ;
    if ~isfield(opts1.train, 'gpus'), opts1.train.gpus = [2]; end;
    
    opts1.train.batchSize    = 256;
    opts1.train.learningRate = [0.01*ones(1,10) 0.005*ones(1,10) 0.0005*ones(1,5)];
    opts1.train.weightDecay  = 0.0005 ;
    opts1.train.numEpochs    = 1;
    opts1.train.gpus         = [2];
    opts1.networkType        = 'dagnn';
    
    opts2.expDir  = fullfile('your\data\path') ;
    trainOpts2.learningRate   = [1*1E-3*ones(1, 10) 1E-4*ones(1, 10) 1E-5*ones(1, 5)] ;
    opts2.train.gpus            = 1 ;
    opts2.train.batchSize       = 256;
    opts2.train.numEpochs       = 1;
    opts2.networkType        = 'dagnn';
    opts2.train.learningRate = [0.01*ones(1,10) 0.005*ones(1,10) 0.0005*ones(1,5)];
    opts2.train.weightDecay  = 0.0005 ;
    opts2.train.gpus         = [2];
    
    
    ConvLayer = [1, 6, 11, 16, 21, 26, 31, 33];
    ratio     = [0,0.25,0.5,0,0.25,0.5,0.75,1,1]; % Tunable parameters, modify the ratio if necessary for your dataset.
    NumEpoch  = 40;
    netlow    = net;
    nethigh   = net;
    
    for i = 1:NumEpoch
        
        if mod(i,2) == 1
            [netlow,info]    = cnn_train_dag(netlow, imdb_low, getBatch_dual_history(opts1), ...
                'expDir', opts1.expDir, ...
                opts1.train, ...
                'val', find(imdb_low.images.set == 2));
            
            opts1.train.numEpochs =  opts1.train.numEpochs + 1;
            for j = 1:numel(ConvLayer)
                weightsize = size(netlow.params(ConvLayer(j)).value);
                nethigh.params(ConvLayer(j)).value(:,:,:,1:round(ratio(j)*weightsize(4))) = netlow.params(ConvLayer(j)).value(:,:,:,1:round(ratio(j)*weightsize(4)));
                if j ~= 8
                    nethigh.params(ConvLayer(j)+1).value(1:round(ratio(j)*weightsize(4)))     = netlow.params(ConvLayer(j)+1).value(1:round(ratio(j)*weightsize(4)));
                end
            end
            
            
        elseif mod(i,2) == 0
            
            [nethigh,info] = cnn_train_dag(nethigh, imdb_high, getBatch_dual_history(opts2), ...
                'expDir', opts2.expDir, ...
                opts2.train, ...
                'val', find(imdb_high.images.set == 2));
            
            opts2.train.numEpochs =  opts2.train.numEpochs + 1;
            for j = 1:numel(ConvLayer)
                weightsize = size(nethigh.params(ConvLayer(j)).value);
                netlow.params(ConvLayer(j)).value(:,:,:,1:round(ratio(j)*weightsize(4))) = nethigh.params(ConvLayer(j)).value(:,:,:,1:round(ratio(j)*weightsize(4)));
                if j ~= 8
                    netlow.params(ConvLayer(j)+1).value(1:round(ratio(j)*weightsize(4)))     = nethigh.params(ConvLayer(j)+1).value(1:round(ratio(j)*weightsize(4)));
                end
            end
        end
        
    end
    
    
end
