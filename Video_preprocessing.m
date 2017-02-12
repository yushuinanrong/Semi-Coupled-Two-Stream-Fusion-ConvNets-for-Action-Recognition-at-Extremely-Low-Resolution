clc;close all
%Author: Jiawei Chen

% For HMDB dataset, load the split index file and video data
% split_Files = dir('E:\Jiawei\Action Recognition\HMDB\split1\*.txt');
% vid_dir     = dir('E:\Jiawei\Action Recognition\HMDB\Data_Org\');
% HMDB_root   = 'E:\Jiawei\Action Recognition\HMDB\Data_Org\';
count       = 1;

labels   = [];
data     = [];
FrameNum = [];
CovSize  = [32,32];
M        = 12; % resize height
N        = 16; % reisize width
set      = [];
vidflag  = 1;

for i = 6
    
    i
    [vidtag,set_info] = textread(strcat('E:\Jiawei\Action Recognition\HMDB\split1\',split_Files(i).name),'%s %d');
    VidFiles          =  dir(strcat(HMDB_root,vid_dir(count).name,'\*.avi'));
   
    
    for k = 1:numel(set_info)
        if set_info(k) == 1 || set_info(k) == 2
            fname=strcat(HMDB_root,vid_dir(count).name,'\',VidFiles(k).name);
            % user tag set to 'myreader1'.
            readerobj = VideoReader(fname);
            vidFrames = read(readerobj);
            numFrames = get(readerobj, 'NumberOfFrames');
            numFrames = numFrames -1;
            FrameNum  = [FrameNum,numFrames];
            
            set_    = set_info(k)*ones(1,numFrames);
            set     = [set set_];
            LB      = ones(2,numFrames);
            LB(1,:) = (count-2)*LB(1,:);
            LB(2,:) = vidflag*LB(2,:);
            labels  = cat(2,labels,LB);
            
            tic
            J = zeros(CovSize(1),CovSize(2),1,numFrames);
            for t=1:numFrames
                
                mov(t).cdata    = vidFrames(:,:,:,t);
                mov(t).colormap = [];
                I               = rgb2gray(mov(t).cdata);
                Intmed          = imresize(I,[M N]);
                J(:,:,:,t)      = imresize(Intmed,CovSize);
            end
            
            % Mean-Variance normalization:
            sigma = std(reshape(J,1,CovSize(1)*CovSize(2)*numFrames));
            avg   = mean(J,4);
            J3    = (J-repmat(avg,[1 1 1 numFrames]))./sigma;
            J3    = J-repmat(avg,[1 1 1 numFrames]);
            data  =  cat(4,data,J3);
            toc
            vidflag = vidflag + 1;
        end
    end
    count = count + 1;
    
end



imdb.images.data    = data;
imdb.images.labels  = labels;
imdb.images.set     = set;
imdb.images.tempLen = FrameNum;
save('your/file/name','imdb')