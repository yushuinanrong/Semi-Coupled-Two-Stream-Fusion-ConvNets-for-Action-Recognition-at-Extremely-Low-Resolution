% Author:Jiawei Chen

% IXMAS dataset
clc;close all
load('your/video/data')
load('your/optical flow/data')

% run  vl_setupnn
% opts.modelType   = 'end2end_fc1fusion_sum' ;
opts.modelType   = 'end2end_fc1fusion_sum_OpticalStack';

for Testpp       = 1:10
    opts.expDir      = fullfile('your/file/path') ;
    
    opts.whitenData  = false;
    opts.contrastNormalization = false;
    opts.networkType = 'dagnn';
    Tempstrach    =  true;
    if Tempstrach == true % for ixmas dataset only
        tempLen = 100*ones(1800,1);
    elseif Tempstrach == false
        tempLen = imdb.images.tempLen;
    end
    
    Ind     = cumsum(tempLen); % video Ind
    Ind2    = reshape(Ind,180,10); % change this line according your data size
    Ind2    = Ind2(180,:);
    set     = ones(1,length(imdb.images.set));
    if Testpp == 1
        set(1:Ind2(1)) = 2;
    else
        set(Ind2(Testpp-1)+1:Ind2(Testpp))=2;
    end
    
    % imdb.images.data   = single(imdb.images.data);
    % imdb.images.labels = single(imdb.images.labels);
    imdb.images.set    = set;
    opts.train = struct() ;
    if ~isfield(opts.train, 'gpus'), opts.train.gpus = [1]; end;
    
    %-----------------------Intialize network--------------------------------
    
    switch opts.modelType
        case 'end2end_fc1fusion_sum'
            net = dagnn_net_end2end_fuse_fc1_ini2();
        case 'end2end_fc1fusion_concate'
            net = dagnn_net_end2end_fuse_concate_fc1_ini();
        case 'end2end_fc1fusion_conv'
            net = dagnn_net_end2end_fuse_conv_fc1_ini2();
        case 'end2end_fc1fusion_concate_OpticalStack'
            net = dagnn_net_end2end_fuse_concate_fc1_opticalstack_ini();
        case 'end2end_fc1fusion_conv_OpticalStack'
            net = dagnn_net_end2end_fuse_conv_fc1_opticalstack_ini();
        case 'end2end_fc1fusion_sum_OpticalStack'
            net = dagnn_net_end2end_fc1fuse_sum_optstack_ini2();
        case 'end2end_lastconv_fusion_conv_OpticalStack'
            net = dagnn_net_end2end_fuse_lastcv_conv_opticalstack_ini();
        case 'end2end_lastconv_fusion_sum_OpticalStack'
            net = dagnn_net_end2end_fuse_lastcv_sum_opticalstack_ini();
    end
    
    switch opts.networkType
        case 'simplenn', trainfn = @cnn_train_IXMAS ;
        case 'dagnn',    trainfn = @cnn_train_dag ;
    end
    
    [net, info] = trainfn(net, imdb, getBatch_dual_history(opts), ...
        'expDir', opts.expDir, ...
        net.meta.trainOpts, ...
        opts.train, ...
        'val', find(imdb.images.set == 2)) ;
end