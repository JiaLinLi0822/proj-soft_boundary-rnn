function shape_prob = makeStimProb(id,varargin)

if isempty(varargin)
    trump = 0;
else
    trump = varargin{1};
end

switch id
    case 1
        stim_ratio = [0 1 2 4 8];
    case 2
        stim_ratio = [0.1 1 2 4 8];
end
    
llr = [9999 9 7 5 3];
llr = llr/10;

sampling_p = stim_ratio./sum(stim_ratio);
p1 = sampling_p.*10.^(llr)./(1+10.^(llr));
pick = find(isnan(p1));
p1(pick) = sampling_p(pick);
p2 = sampling_p-p1;
pp1 = [p1,fliplr(p2)];
pp2 = fliplr(pp1);

pR = pp2;

if trump
    shape_prob = [pR(1:5), 0, 0, pR(6:10)];
else
    shape_prob = [pR(2:5), 0, 0, pR(6:9)];
end