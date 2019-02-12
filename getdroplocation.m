function droploc = getdroplocation( index )
%GETDROPLOCATION Summary of this function goes here
%   Detailed explanation goes here


% pick up/ drop off locations
pudo=[1,5,21,24];
if (index > length(pudo)), droploc=-1; else droploc=pudo(index); end

end

