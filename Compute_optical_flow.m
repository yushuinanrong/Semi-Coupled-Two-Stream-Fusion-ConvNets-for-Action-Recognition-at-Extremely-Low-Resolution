addpath('mex');
% Author: Jiawei Chen
% Revised based on Chi Liu's code : https://people.csail.mit.edu/celiu/OpticalFlow/

a = load('your\data\path');% load your video data
[w,h,c,num] = size(a.images.data);
Opt_img     = (zeros(32,32,3,num));

for i = 1:num-1
    i
    im1 = a.images.data(:,:,:,i);
    im2 = a.images.data(:,:,:,i+1);
    % set optical flow parameters (see Coarse2FineTwoFrames.m for the definition of the parameters)
    alpha = 0.012;
    ratio = 0.75;
    minWidth = 20;
    nOuterFPIterations = 7;
    nInnerFPIterations = 1;
    nSORIterations = 30;
    
    para = [alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations];
    
    % this is the core part of calling the mexed dll file for computing optical flow
    % it also returns the time that is needed for two-frame estimation
    tic;
    [vx,vy,warpI2] = Coarse2FineTwoFrames(im1,im2,para);
    toc
    
    % figure;imshow(im1);figure;imshow(warpI2);
    
    
    
    % output gif
    clear volume;
    volume(:,:,:,1) = im1;
    volume(:,:,:,2) = im2;
    % if exist('output','dir')~=7
    %     mkdir('output');
    % end
    % frame2gif(volume,fullfile('output',[example '_input.gif']));
    volume(:,:,:,2) = warpI2;
    % frame2gif(volume,fullfile('output',[example '_warp.gif']));
    
    
    % visualize flow field
    clear flow;
    flow(:,:,1) = vx;
    flow(:,:,2) = vy;
    imflow = flowToColor(flow);
    
    Opt_img(:,:,:,i) = imflow;
%     
    hold on
    imshow(imflow);
    shg
    pause(0.1)

    
end

imdb.images.data   = (Opt_img);
imdb.images.labels = a.images.labels;
imdb.images.set    = a.images.set;

save('your/data/path','imdb')

