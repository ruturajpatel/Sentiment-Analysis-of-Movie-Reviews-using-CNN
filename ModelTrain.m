%% CMPT-741 Project 
% Sentiment Analysis base on Convolutional Neural Network Model
% author: Ruturaj Patel, Aakash Moghariya
% date: 04/12/2016
%matconvnet-1.0-beta23/matlab/vl_setupnn
clear; clc;

%% Section 1: preparation before training
% section 1.1 read file 'train.txt', load data and vocabulary by using function read_data()
[x_data,test_data] = read_data();

fprintf('read data loaded\n');

gloveVector = load('wordvector_with_pad.mat');
wordVector = gloveVector.wordVector;

fprintf('vector data loaded\n');

eta = 0.5;
prediction_count = 0;

d = 300;

filter_size = [2,3,4];
no_filter = 6;
filter_len = length(filter_size);
wConv = cell(filter_len,1);
bConv = cell(filter_len,1);
weightDerivative = cell(filter_len,1);
baisDerivative = cell(filter_len,1);

convDer = cell(filter_len,1);

for i = 1: filter_len

	f = filter_size(i);
	
	wConv{i} = normrnd(0,0.1,[f,d, 1, no_filter]);
	bConv{i} = zeros(no_filter,1);
end

total_filters = filter_len * no_filter;

classes = 2;

wOut = normrnd(0,0.1,[total_filters,classes]);

bOut = zeros(classes,1);

%% Section 2: training
% Note:
% you may need the resouces [2-4] in the project description.
% you may need the follow MatConvNet functions:
%       vl_nnconv(), vl_nnpool(), vl_nnrelu(), vl_nnconcat(), and vl_nnloss()

% for each example in train.txt do
% section 2.1 forward propagation and compute the loss

pool_res = cell(1, filter_len);
cache = cell(1, filter_len);

loss = cell(1, length(x_data));
testLoss = cell(1, length(test_data));



ep = 1;
while ep > 0
	count = 0;
    prediction_count_train = 0;
	for i=1:length(x_data)

		words_arr = x_data{i,2};
		word_indexes = cell(length(words_arr),1);
		X = [];
		
        if length(words_arr) <  filter_len
            padlist=[];
            for padindex = length(words_arr)+1:filter_len
                padlist = [padlist '<pad>'];
            end
            words_arr = [words_arr padlist];
        end
        
        
		for j=1:length(words_arr)
			curr_word = char(words_arr(j));

			if isKey(wordVector,curr_word)
				X = [X ; wordVector(curr_word)];
			else
				count = count + 1;
				rSample = normrnd(0,0.1,[1,d]);
				wordVector(curr_word) = rSample;
				X = [X;rSample];
			end
		end

		[pool_res,cache] = neural_net(X,filter_len,wConv, bConv,cache,pool_res);

		fullCon = vl_nnconcat(pool_res,3);
		reshapedFC = reshape(fullCon,total_filters,1);
		wOut = reshape(wOut,total_filters,1,1,classes);

		output = vl_nnconv(reshapedFC, wOut,bOut);

		targetLabel = x_data{i,3};
		targetLabel = targetLabel + 1;

		loss{i} = vl_nnloss(output,targetLabel);

		lossDer = 1;
		outputDer = vl_nnloss(output,targetLabel,lossDer);

		[dataDer, weightsDer, biasDer] = vl_nnconv(reshapedFC, wOut, bOut, outputDer) ;
		reshapedDataDer = reshape(dataDer,1,1,total_filters);

		fullcDer = vl_nnconcat(pool_res,3,reshapedDataDer);

		for k = 1:filter_len
            if k < length(X(:,1))

				sizes = size(cache{1,k});
				pollDer= vl_nnpool(cache{2,k},[sizes(1),1],fullcDer{1,k});
				reluDer = vl_nnrelu(cache{1,k},pollDer);
				[convDer{k},weightDerivative{k},baisDerivative{k}] = vl_nnconv(X,wConv{k},bConv{k},reluDer);
				wConv{k}= wConv{k} - weightDerivative{k}*eta;
				bConv{k}= bConv{k} - baisDerivative{k}*eta;

				for j=1:length(words_arr)
					curr_word = char(words_arr(j));
					wordVector(curr_word) = wordVector(curr_word) - eta * convDer{k}(j, :);
				end
			
            end

		end

		[~,pred]=max(output);
		
		if pred == targetLabel
			prediction_count_train = prediction_count_train + 1;
		end
	end

	fprintf('\n train prediction total ');
	fprintf('%i',prediction_count_train);
	fprintf('\n train Accuracy ');
	fprintf('%i \n \n \n ',(prediction_count_train/length(x_data)));
    
    graphval{ep,1} = ep;
	graphval{ep,2} = mean([loss{:}]);
	fprintf('loss is %f  \n',mean([loss{:}]));
	fprintf('done with epoch %i \n',ep);
	fprintf('done with eta %d \n',eta);
	
	if ep > 7 && ep <= 20
		eta = 0.1;
	elseif ep > 20 && ep <=30
		eta = 0.01;
	elseif ep > 30 && ep <=50
		eta = 0.001;
	end

	if (ep > 1) && ((graphval{ep-1,2} - graphval{ep,2}) <= 0.0001)
		fprintf('Total Eta Completed === %i \n',eta);
		break;
	end
	
	ep = ep + 1;
end

fprintf('finish training model\n');

for i=1:length(test_data)
	words_arr = test_data{i,2};
	word_indexes = cell(length(words_arr),1);
	X = [];
    
    
    
	for j=1:length(words_arr)
		curr_word = char(words_arr(j));
		
		if isKey(wordVector,curr_word)
			X = [X ; wordVector(curr_word)];
		else
			rSample = normrnd(0,0.1,[1,d]);
			X = [X ; rSample];
		end
	end

	[pool_res,cache] = neural_net(X,filter_len,wConv, bConv,cache,pool_res);

	fullCon = vl_nnconcat(pool_res,3);
	reshapedFC = reshape(fullCon,total_filters,1);
	wOut = reshape(wOut,total_filters,1,1,classes);

	output = vl_nnconv(reshapedFC, wOut,bOut);

	targetLabel = test_data{i,3};
	targetLabel = targetLabel+1;

	[~,pred]=max(output);
	
	if pred == targetLabel
		prediction_count = prediction_count + 1;
	end
end


fprintf('\n prediction total ');
fprintf('%i',prediction_count);
fprintf('\n Accuracy ');
fprintf('%i',(prediction_count/length(test_data)));

save('weights.mat','wConv','bConv','wOut','bOut');