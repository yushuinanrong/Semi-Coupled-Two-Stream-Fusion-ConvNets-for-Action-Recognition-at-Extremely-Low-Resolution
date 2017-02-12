% -------------------------------------------------------------------------
function inputs = getDagNNBatch_dual_history(opts, imdb, batch)
% Author: Jiawei Chen

% -------------------------------------------------------------------------
images_src_1 = imdb.images.data_src1(:,:,:,batch) ;
% images_src_2 = imdb.images.data_src2(:,:,:,batch) ;
L = 5;
[w,h,c,num]        = size(imdb.images.data_src2);
images_src_2       = zeros(w,h,c*(2*L+1),numel(batch));
for i = 1:numel(batch)
    for j = 1:2*L+1
        ind = batch(i)-L+j-1;
        if ind < 1
            ind = 1;
        elseif ind >num
            ind = num;
        end
        images_src_2(:,:,(j-1)*c+1:j*c,i) = imdb.images.data_src2(:,:,:,ind);
    end
    images_src_2(:,:,:,i) =  images_src_2(:,:,:,i) - repmat(mean(images_src_2(:,:,:,i),3),[1,1,size(images_src_2,3),1]);
end
images_src_2 = single(images_src_2);


labels = imdb.images.labels(1,batch) ;
if rand > 0.5
    images_src_1 =fliplr(images_src_1 ) ; 
    images_src_2 =fliplr(images_src_2 ) ; 
end
if opts.numGpus > 0
  images_src_1 = gpuArray(images_src_1 ) ;
  images_src_2 = gpuArray(images_src_2 ) ;
end
% **********************************************
% Define the inputs cell-array to the DAG
% **********************************************
inputs = {'input_1', images_src_1 , 'label', labels, 'input_2', images_src_2, 'label', labels} ;