% Merging .mat files into 1 file, only containing variables in array form
% Ref: https://uk.mathworks.com/matlabcentral/answers/318025-merging-mat-files-into-1-file-only-containing-variables-in-array-form
clear;clc

current_dir = pwd;
dir_slash = strfind(current_dir,'/');
system_config = current_dir(dir_slash(end)+1:end);
train_dir = [current_dir '/raw/train'];
test_dir = [current_dir '/raw/test'];
merge_dir = [current_dir, '/merge'];

pmin = -3;
pmax = 5;

filename_train = dir(train_dir);
filename_test = dir(test_dir);
filename_array_train = {};
filename_array_test = {};
for idx_file = 1:length(filename_train)
    filename_array_train{end+1} = filename_train(idx_file).name;
    filename_array_test{end+1} = filename_test(idx_file).name;
end

filename_array_train = filename_array_train(3:end);
filename_array_test = filename_array_test(3:end);

for plch = pmin:pmax
    plch_str = ['_'  num2str(plch) '_Pdbm'];
    loc = find(~cellfun(@isempty,strfind(filename_array_test, plch_str)));
    train_file_path = [train_dir '/' filename_array_train{loc}];
    test_file_path = [test_dir '/' filename_array_test{loc}];
    
    % main code
    FileList = {train_file_path, test_file_path};
    data = struct();
    for iFile = 1:numel(FileList)               % Loop over found files
        Data   = load(FileList{iFile});
        Fields = fieldnames(Data);
        for iField = 1:numel(Fields)              % Loop over fields of current file
            aField = Fields{iField};
            if isfield(data, aField)             % Attach new data:
                %allData.(aField) = [allData.(aField), Data.(aField)];
                
                % [EDITED]
                % The orientation depends on the sizes of the fields. There is no
                % general method here, so maybe it is needed to concatenate
                % vertically:
                data.(aField) = [data.(aField); Data.(aField)];
                % Or in general with suiting value for [dim]:
                % allData.(aField) = cat(dim, allData.(aField), Data.(aField));
            else
                data.(aField) = Data.(aField);
            end
        end
    end
    merge_file_name = [system_config plch_str '.mat'];
    save(fullfile(merge_dir, merge_file_name), '-struct', 'data');
    
end


