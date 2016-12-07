%% CMPT-741 Project 
% Test trained Convoluted Nerual Network Model for Sentimental Analysis.
% author: Ruturaj Patel, Aakash Moghariya
% date: 04/12/2016

[test_data] = read_test_data();

fprintf('read data loaded\n');
vector_file = load('wordvector_with_pad.mat');
wordVector = vector_file.wordVector;

fprintf('vector data loaded\n');

prediction_count = 0;
d = 300;
write_file = input('Enter the file name to save: ','s');
fileID = fopen(write_file,'w');
fprintf(fileID,'%s::%s\r\n','id','label');
fclose(fileID);

filter_size = [2,3,4];
no_filter = 6;

filter_len = length(filter_size);
trainedWeights = load('weights_with_pad.mat');
wConv = trainedWeights.wConv;
bConv = trainedWeights.bConv;
wOut = trainedWeights.wOut;
bOut = trainedWeights.bOut;

%disp(size(bOut));
%disp(size(wOut));
%% Section 2: training
% Note:
% you may need the resouces [2-4] in the project description.
% you may need the follow MatConvNet functions:
%       vl_nnconv(), vl_nnpool(), vl_nnrelu(), vl_nnconcat(), and vl_nnloss()
total_filters = filter_len * no_filter;
classes = 2;

pool_res = cell(1, filter_len);
cache = cell(1, filter_len);

testLoss = cell(1, length(test_data));


fprintf('finish training model\n');
fileID = fopen(write_file,'a');

for i=1:length(test_data)
	words_arr = test_data{i,2};
	word_indexes = cell(length(words_arr),1);
	X = [];

	for j=1:length(words_arr)
		curr_word = char(words_arr(j));
        
        if length(words_arr) <  filter_len
            padlist=[];
            for padindex = length(words_arr)+1:filter_len
                padlist = [padlist '<pad>'];
            end
            words_arr = [words_arr padlist];
        end
		
        
		if isKey(wordVector,curr_word)
			X = [X ; wordVector(curr_word)];
		else
			rSample = normrnd(0,0.1,[1,d]);
			X = [X ; rSample];
        end
        
        
	end

	[pool_res,cache] = neural_net(X,filter_len,wConv, bConv,cache,pool_res);
	fullConnected = vl_nnconcat(pool_res,3);
	reshapedFC = reshape(fullConnected,total_filters,1);

	wOut = reshape(wOut,total_filters,1,1,classes);
	output = vl_nnconv(reshapedFC, wOut,bOut);

	sid = test_data{i,1};
	[~,pred]=max(output);

	fprintf(fileID,'%d::%d\r\n',int16(sid),int16(pred-1));
end

fprintf('\n\ndone');
fclose(fileID);
