[heads,sequences] = fastaread('PDNA-224.fasta');
N = length(heads);

DBsites = cell(1,N);
fid = fopen('PDNA-224-binding-sites.txt','r');
i = 1;
while( i <= N)
    tline = fgetl(fid);
    tline = fgetl(fid);

    m = length(sequences{i});
    sites = zeros(1,m);
    str = strtrim(tline);
    arry = split(str);
    for j = 1 : size(arry,1)
        k = str2double(arry(j));
        sites(k) = 1;
    end
    DBsites{i} = sites;
    i = i + 1;
end
save( 'PDNA-224.mat', 'heads', 'sequences', 'DBsites');