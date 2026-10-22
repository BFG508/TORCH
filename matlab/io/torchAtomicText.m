function torchAtomicText(path,text)
% Replace a text file only after the complete temporary file has been closed.
parent=fileparts(path);if isempty(parent),parent=pwd;end;if ~exist(parent,'dir'),mkdir(parent);end
temporary=tempname(parent);fid=fopen(temporary,'w','n','UTF-8');assert(fid>=0,'Cannot create output');
cleanup=onCleanup(@()cleanupFile(fid,temporary));fprintf(fid,'%s',text);fclose(fid);
[ok,message]=movefile(temporary,path,'f');assert(ok,message);
end
function cleanupFile(fid,path)
try, fclose(fid); catch, end
if exist(path,'file'),delete(path);end
end
