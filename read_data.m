function [data,test_data] = read_data()
% CMPT-741 Project
% Reading training data and building vocabulary.
% NOTE: reading testing data is similar, but no need to build the vocabulary.
%
% return:
%       data(cell), 1st column -> sentence id, 2nd column -> words, 3rd column -> label
%       wordMap(Map), contains all words and their index, get word index by calling wordMap(word)
	
	headLine = true;
	separater = '::';

	words = [];
	fulldata = cell(6000,3);

	fid = fopen('train.txt', 'r');
	line = fgets(fid);

	ind = 1;
	
	count = 1;

	while ischar(line)
		
		if headLine
			line = fgets(fid);
			headLine = false;
		end
		
		attrs = strsplit(line, separater);
		sid = str2double(attrs{1});

		s = attrs{2};
		w = strsplit(s);
		words = [words w];

		y = str2double(attrs{3});

		% save data
		fulldata{ind, 1} = sid;
		fulldata{ind, 2} = w;
		fulldata{ind, 3} = y;
		ind = ind + 1;
		
		line = fgets(fid);

		count = count + 1;
	end

	fulldata = fulldata(randperm(size(fulldata,1)),:);

	test_data = fulldata(1:700,:);

	data = fulldata(801:6000,:);
	data = data(randperm(size(data,1)),:);

	
	fprintf('finish loading data and vocabulary\n');