function info = neuralSPRT(id,fig_switch_beh,fig_switch_phy)

% neuralSPRT reproduces figures and stats in:
% 
% Kira, S., Yang, T., Shadlen, M.N. (2015) Neuron
% A neural implementation of Walds sequential probability ratio test.
% 
% neuralSPRT(id, fig_switch_beh, fig_switch_phy) reproduces figures and 
% stats for the monkey specified by id (1: monkey E, 2: monkey J). 
% 
% To reproduce figures in the paper, choose appropriate numbers for 
% fig_switch_beh (behavior) and fig_switch_phy (physiology) from 
% the following list.
% 
% For example, neuralSPRT(1, 1, []) produces Figure 1A for monkey E.
%
% fig_switch:
% 1: Fig. 2A -- RT histogram
% 2: Fig. 2B -- Effect of a single shape presented at a specific timing
% 3: Fig. 2C -- N* histogram
% 4: Fig. 2D -- psychometric function
% 5: Fig. 2E -- Cumulative evidence at the end of decision (also Fig. S1)
% 6: Fig. 2F -- Subjective weight
%
% fig_switch_phy:
% 1: Fig. 3A -- LIP activity accompanying decision formation (PSTH + insets)
% 2: Fig. 3B -- Change in FR (delta FR) (also Fig. S2B)
% 3: Fig. 4A -- LIP activity accompanying decision termination (PSTH)
% 4: Fig. 4B -- Change in FR associated with N*th shapes

close all
rng('default')

% set default properties for Figures
set(0,'defaultaxesfontsize',24);
set(0,'defaulttextfontsize',24);
set(0,'defaultaxesfontweight','bold');
set(0,'defaulttextfontweight','bold');
set(0,'defaultaxestickdir','out');
set(0,'defaultaxesbox','off');
set(0,'defaultFigureColor','w');

switch id
    case 1
        Monkey = 'E';
        tau_s = 200; % (ms) 'sensory' delay from the shape onset until it gets reflected in FR
    case 2
        Monkey = 'J';
        tau_s = 130; % (ms) 'sensory' delay from the shape onset until it gets reflected in FR
end

% load data file
load(['data',Monkey])

% Removing unnecessary data
info = rmfield(info,'FM');
info = rmfield(info,'SpM');

% In info.tsMtrx, the first, the first column contains time stamps of 
% events and the second column contains 'event codes'. 'ecode790sk.m' 
% contains a look-up table for the event codes.
% For example, ecode 601 (= E_SPIKE) in tsMtrx indicates that a spike 
% occured at the time indicated by its time stamp in the same row.

ecodeRTshape;

info.tau_s = tau_s;
% info.home_dir = home_dir;
% info.data_dir = data_dir;

fprintf('TM size: %d x %d\n\n',size(info.TM,1),size(info.TM,2))

% Removing unnecessary data
% info = rmfield(info,'elMtrx');
% info = rmfield(info,'RM');
% info = rmfield(info,'LM');

if isfield(info,'adMtrx')
    info = rmfield(info,'adMtrx');
end

info.fig = 1;
if ~isempty(fig_switch_beh)
    info = neuralSPRT_BEH(info,id,fig_switch_beh); % Behavioral analyses
end
if ~isempty(fig_switch_phy)
    info = neuralSPRT_PHY(info,id,fig_switch_phy); % Physiological analyses
end
    
return