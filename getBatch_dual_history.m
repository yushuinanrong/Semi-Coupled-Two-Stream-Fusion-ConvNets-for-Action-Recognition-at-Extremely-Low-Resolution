function fn = getBatch_dual_history(opts)
% -------------------------------------------------------------------------
switch lower(opts.networkType)
    case 'simplenn'
        fn = @(x,y) getSimpleNNBatch(x,y) ;
    case 'dagnn'
        bopts = struct('numGpus', numel(opts.train.gpus)) ;
        fn = @(x,y) getDagNNBatch_dual_history(bopts,x,y) ;
end
