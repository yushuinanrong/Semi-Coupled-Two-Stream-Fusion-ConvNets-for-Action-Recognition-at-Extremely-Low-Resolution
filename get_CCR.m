% Author: Jiawei Chen
% Compute CCR value
clc;close all

load('your\testdata')
CCR           = [];
Tempstrach    =  true; % IXMAS only
L = 5;
for ratio = 9
    for Testpp = 1:10
        
        if Tempstrach == true
            tempLen = 100*ones(1800,1);
        elseif Tempstrach == false
            tempLen = imdb.images.tempLen;
        end
        Ind     = cumsum(tempLen); % video Ind
        Ind2    = reshape(Ind,180,10);
        Ind2    = Ind2(180,:);
        set     = ones(1,length(imdb.images.set));
        if Testpp == 1
            set(1:Ind2(1)) = 2;
        else
            set(Ind2(Testpp-1)+1:Ind2(Testpp))=2;
        end
        imdb.images.set    = set;
        
        %------------------Load net-----------------------------------------
        % load the pre-trained CNN
        net = ('your\trained_net'); 
        net = net.net;
        net = dagnn.DagNN.loadobj(net);
        net.mode = 'test' ;
        
        %--------------------Load test images-----------------------------
        im_1_all = imdb.images.data_src1(:,:,:,set ==2);
        im_2_all = imdb.images.data_src2(:,:,:,set ==2);
        Ind_set2 = find(imdb.images.set==2);
        %----------------------Evaluation-----------------------------------
        Best_vec           = zeros(numel(find(imdb.images.set ==2)),1);
        [w,h,c,num]        = size(imdb.images.data_src2);

        scores_mat = [];
        apt = 1;
        for i = 1:numel(find(imdb.images.set ==2))
            i
            im_1 = im_1_all(:,:,:,i);
            im_2 = zeros(w,h,c*(2*L+1),1);
            for j = 1:2*L+1
                ind = Ind_set2(i)-apt*L+j*apt-1*apt;
                if ind < 1
                    ind = 1;
                elseif ind >num
                    ind = num;
                end
                im_2(:,:,(j-1)*c+1:j*c,1) = imdb.images.data_src2(:,:,:,ind);
            end
            im_2(:,:,:,1) =  im_2(:,:,:,1) - repmat(mean(im_2(:,:,:,1),3),[1,1,size(im_2,3),1]);
            im_2 = single(im_2);

            % run the CNN
            net.eval({'input_1', im_1,'input_2',im_2}) ;
            % obtain the CNN otuput
            scores = net.vars(net.getVarIndex('pred')).value ;
            scores = squeeze(gather(scores)) ;
            % show the classification results
            [bestScore, best] = max(scores) ;
            Best_vec(i) = best;
            scores_mat = cat(2,scores_mat,scores);
        end
        
        Label       = imdb.images.labels;
        Vid_ind     =  unique(Label(2,set==2));
        GrondLB     = Label(:,set==2);
        Pred        = [Best_vec';GrondLB(2,:)];
        
        Performance = [];
        for i = 1:length(Vid_ind)
            
            Truth_ind1  = find(GrondLB(2,:) == Vid_ind(i));
            Truth       = mode(GrondLB(1,Truth_ind1)); 
            Pred_ind1   = find(Pred(2,:) == Vid_ind(i));
            Pred_scores = scores_mat(:,Pred_ind1);
            [bestScore, prediction] = max(mean(Pred_scores,2)) ;
            Performance = [Performance,isequal(Truth,prediction)];
            
        end
        CCR = [CCR, sum(Performance)/length(Performance)]
    end
end
