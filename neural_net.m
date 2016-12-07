%% CMPT-741 Project 
% Neural Net Architecture Generator
% author: Ruturaj Patel, Aakash Moghariya
% date: 04/12/2016

function [pool_res,cache] = neural_net(X,filter_len,wConv, bConv,cache,pool_res)
	
	for k = 1:filter_len
        
       
		if(k < length(X(:,1)))
        
			% Creates Convolution and ReLU Layer
			conv = vl_nnconv(X,wConv{k},bConv{k});
			relu = vl_nnrelu(conv);
			sizes = size(conv);

			% Creates pooling layer.
			pool = vl_nnpool(relu,[sizes(1),1]);
			
			% To store convolution and ReLU layer values.
			cache{2,k} = relu;
			cache{1,k} = conv;
			
			% Crates pool vector.
			pool_res{k} = pool;
		end

	end