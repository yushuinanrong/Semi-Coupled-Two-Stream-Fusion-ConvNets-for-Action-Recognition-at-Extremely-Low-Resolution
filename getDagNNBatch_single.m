function inputs = getDagNNBatch_single(opts, imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if rand > 0.5, images=fliplr(images) ; end
if opts.numGpus > 0
    images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ;