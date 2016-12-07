%save('read_data.mat','x_data','test_data','wordmap');
[x_data,test_data] = read_data();
%[mywordmap] = test_vector();
fprintf('read data loaded');
vector_file = load('wordvector_with_pad.mat');
%vector_file = load('train_map.mat');
mywordmap = vector_file.wordVector;
%mywordmap = vector_file.trainmap;

fprintf('vector data loaded');

prediction_count = 0;
d = 300;

filter_size = [2,3,4];
no_filter = 6;
minibatch = 10;
filter_len = length(filter_size);
myweights = load('weights_with_pad.mat');
wConv = myweights.wConv;
bConv = myweights.bConv;
wOut = myweights.wOut;
bOut = myweights.bOut;

%disp(size(bOut));
%disp(size(wOut));
%% Section 2: training
% Note: 
% you may need the resouces [2-4] in the project description.
% you may need the follow MatConvNet functions: 
%       vl_nnconv(), vl_nnpool(), vl_nnrelu(), vl_nnconcat(), and vl_nnloss()
total_filters = filter_len * no_filter;
%fprintf('total filters');
%disp(total_filters);

classes = 2;

% for each example in train.txt do
% section 2.1 forward propagation and compute the loss
% TODO: your code

pool_res = cell(1, filter_len);
cache = cell(1, filter_len);

loss = cell(1, length(x_data));
myloss = cell(1, length(test_data));
epoch = 50;
graphval = cell(epoch, 2);

fprintf('finish training model\n');
for i=1:length(test_data)

%    [word_indexes] = word_to_vec(test_data,i,wordmap);
%     
%     X = rSample(word_indexes,:);
        words_arr = test_data{i,2};
             word_indexes = cell(length(words_arr),1);
            X = [];
          for j=1:length(words_arr)
                curr_word = char(words_arr(j));
                if isKey(mywordmap,curr_word)
                    
                    X = [X ; mywordmap(curr_word)];
                else
                    rSample = normrnd(0,0.1,[1,d]);
                    X = [X ; rSample];
                end
          end
      
    [pool_res,cache] = neural_net(X,filter_len,wConv, bConv,cache,pool_res);
        
    z1 = vl_nnconcat(pool_res,3);
    
    
    z = reshape(z1,total_filters,1);
    
    wOut = reshape(wOut,total_filters,1,1,classes);

    o = vl_nnconv(z, wOut,bOut);
   
    y = test_data{i,3};
    y = y+1;
   
    
    [~,pred]=max(o);
    if pred == y
        prediction_count = prediction_count + 1;
    end
   

end

fprintf('\n prediction total ');
fprintf('%i',prediction_count);
fprintf('\n Accuracy ');
fprintf('%i',(prediction_count/length(test_data)));