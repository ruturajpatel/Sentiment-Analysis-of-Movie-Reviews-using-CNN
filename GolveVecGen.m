%% CMPT-741 Project  
% Glove Vector Generator for Trainig the Model
% author: Ruturaj Patel, Aakash Moghariya
% date: 04/12/2016

fprintf('Vectors generation started\n');
inputFile = 'vectors/glove.6B.300d.txt';

fileIterator = fopen(inputFile);
singleLine = fgets(fileIterator);
separater = ' ';

index = 1;
map = containers.Map('KeyType','char','ValueType','any');

% Converts each line to a word vector.
while ischar(singleLine)

	% Splits line and creates wordVector. Stores the wordVector into a variable.
	items = strsplit(singleLine, separater);
	w = items{1} ;
	wordVector = str2double(items(2:length(items)));
	map(w) = wordVector;

	% Gets next line from the file and increments index.
	singleLine = fgets(fileIterator);
	index = index + 1;
	fprintf('%d \n',index);
end

% Save the word vector to a file.
save('vectors_300_map.mat','map');

fprintf('Vectors generated\n');