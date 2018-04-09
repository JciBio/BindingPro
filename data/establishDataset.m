% rawDataFile: include proteins' information: heads, sequences, DNA-binding
%              sites
% datasetFile: the file name of generating dataset file 
% k: the value of sliding window size
function  establishDataset(rawDataFile, datasetFile,k)
    load(rawDataFile);
    N = length(heads);
    m = 1;
    for i = 1 : N
        slen = length(sequences{i});
        ss = sequences{i};
        for j = 1 : slen
            if ( j < k+1 )
                seq = strcat( ss( end-k+j:end), ss(1: j+k));
            else if ( j > slen - k )
                    seq = strcat( ss( j-k:j-1),ss(j:end), ss(1: k-end+j));
                else
                    seq = strcat( ss( j-k:j-1),ss(j:j+k));
                end
            end
            seqs{m} = seq;
            target{m} = DBsites{i}(j);
            m = m + 1;
        end
    end
    save(datasetFile, 'seqs', 'target');