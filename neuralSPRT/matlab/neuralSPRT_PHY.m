function info = neuralSPRT_PHY(info,id,fig_switch)

% fig_switch_phy:
% 1: Fig. 3A -- LIP activity accompanying decision formation (PSTH + insets)
% 2: Fig. 3B -- Change in FR (delta FR) (also Fig. S2B)
% 3: Fig. 4A -- LIP activity accompanying decision termination (PSTH)
% 4: Fig. 4B -- Change in FR associated with N*th shapes

fig_temp = fig_switch;
fig_switch = zeros(1,7);
for ai = 1:length(fig_temp);
    fig_switch(fig_temp(ai))=1;
end

%%
% trial matrix (TM) contains columns with following information:
% #col 
% 1: reward-assigned target (1:left, 2:right)
% 2: monkey's choice (1:choice B, 2: choice A)
% 3: RT from the first shape onset to the saccade
% 4: a last stim duration before the saccade (sac_latency)
% 5: target x position
% 6: target y position
% 7: RF side (1:left, 2:right)
% 8: Tin or Tout (1:Tin, 2:Tout)
% 9: Target color configuration (Joey only, 1: Green left, 2: Green right)
% 10: Chosen target color (Joey only, 1: Green, 2: Red)
% 11-30: shape index (2-9)

switch id
    case 1 % monkey E
        Pick_RF_flag = 0;
        t_div = 200; %(ms) sensory delay (tau_s)
        lastAccumShape2Sac = 270; %(ms) non-decision time
        stim_int = 250;
    case 2 % monkey J
        Pick_RF_flag = 1;
        t_div = 130; %(ms) sensory delay (tau_s)
        lastAccumShape2Sac = 180; %(ms) non-decision time
        stim_int = 270;
end

%% pre-processing the trial data
    
total_shape = 8;
shape_offset = 2;

fig = info.fig;

TM = info.TM;
KM = info.KM;
FRgain = info.FRgain;

popID = info.popID;

if id==2
    pick = false(size(TM,1),1);
    if Pick_RF_flag % select Red in RF trials only
        if 1
            % red in RF
            pick = logical(TM(:,9)==1) | pick;
        end
        if 0
            % green in RF
            pick = logical(TM(:,9)==2) | pick;
        end
    end
    
    shapeMtrx = TM(:,11:30)+1;
        % Excluding trials with trump shapes for monkey J
    pick = ~logical(sum(shapeMtrx==1|shapeMtrx==2,2)) & pick;
    TM = TM(pick,:);
    KM = KM(pick,:);
    if isfield(info,'num_cell')
        popID = popID(pick);
    end
end

trueDeciWOE = [9999 -9999 9 -9 7 -7 5 -5 3 -3 nan];
shapeMtrx = TM(:,11:30)+1;
frMtrx{1} = TM(:,31:50);
num_shown = sum(isfinite(shapeMtrx),2);
rew_targ = TM(:,1);
if id==2
    % 1:Green, 2:Red
    rew_color = (rew_targ~=TM(:,9))+1;
end
choice = TM(:,2);
RT = TM(:,3);
correct = TM(:,1)==TM(:,2);
T_in_out = TM(:,8);

rawShapeMtrx = shapeMtrx;
rawShapeMtrx(isnan(rawShapeMtrx))=11;
rawLLR{1} = trueDeciWOE(rawShapeMtrx);
rawFrMtrx{1} = frMtrx{1};

    % Generating random order
    % reset the random number seed to generate the same results every time.
rng(1);
rand_trial_order = randperm(size(TM,1));
rng('shuffle');

num_accum = num_shown;

    % Set lastAccumShape2Sac to 0 to include all the observed shapes 
    % (as opposed to accumulated shapes)
t_cutoff = lastAccumShape2Sac*ones(length(num_shown),1);
t_cutoff = t_cutoff-TM(:,4);
pick = t_cutoff>0;
num_accum(pick) = num_accum(pick)-1;
while any(t_cutoff>0)
    t_cutoff = t_cutoff-stim_int;
    pick = t_cutoff>0;
    num_accum(pick) = num_accum(pick)-1;
end

% num_accum = ceil((RT-lastAccumShape2Sac)/270);

for i = 1:size(TM,1)
    if num_accum(i) < 0
        num_accum(i) = 0;
    end
    shapeMtrx(i,num_accum(i)+1:num_shown(i)) = nan;
    frMtrx{1}(i,num_accum(i)+1:num_shown(i)) = nan;
end

shapeMtrx(isnan(shapeMtrx))=11;
LLR{1} = trueDeciWOE(shapeMtrx);

fprintf('create woe matrix with SWOE\n\n');

temp = [TM(:,51),frMtrx{1}];
deltaFrMtrx{1} = diff(temp,1,2);
temp = [TM(:,51),rawFrMtrx{1}];
rawDeltaFrMtrx{1} = diff(temp,1,2);

    % changing the sign of WOE so that the positive WOE favors Tin
    % assuming the monkey assigned the weights symmetrically
    % to positive and negative shapes.
switch id
    case 1
        pick = logical(TM(:,7)==1);
    case 2
        pick = logical(TM(:,9)==2);
end

LLR{1}(pick,:) = -LLR{1}(pick,:);
cumLLR{1} = cumsum(LLR{1},2);

rawLLR{1}(pick,:) = -rawLLR{1}(pick,:);
rawCumLLR{1} = cumsum(rawLLR{1},2);
    
    % preallocation
LLR{2} = nan(size(LLR{1}));
cumLLR{2} = nan(size(cumLLR{1}));
frMtrx{2} = nan(size(frMtrx{1}));
deltaFrMtrx{2} = nan(size(deltaFrMtrx{1}));

rawLLR{2} = nan(size(rawLLR{1}));
rawCumLLR{2} = nan(size(rawCumLLR{1}));
rawFrMtrx{2} = nan(size(rawFrMtrx{1}));
rawDeltaFrMtrx{2} = nan(size(rawDeltaFrMtrx{1}));

    % For backward analyses
for ci = 1:20
    num_shift = 20-ci;
    % dealing with accumulated shapes
    pick = logical(num_accum==ci); 
    LLR{2}(pick,:) = fliplr(circshift(LLR{1}(pick,:),[0,num_shift]));
    cumLLR{2}(pick,:) = fliplr(circshift(cumLLR{1}(pick,:),[0,num_shift]));
    frMtrx{2}(pick,:) = fliplr(circshift(frMtrx{1}(pick,:),[0,num_shift]));
    deltaFrMtrx{2}(pick,:) = fliplr(circshift(deltaFrMtrx{1}(pick,:),[0,num_shift]));

    % dealing with shown shapes
    pick = logical(num_shown==ci);
    rawLLR{2}(pick,:) = fliplr(circshift(rawLLR{1}(pick,:),[0,num_shift]));
    rawCumLLR{2}(pick,:) = fliplr(circshift(rawCumLLR{1}(pick,:),[0,num_shift]));
    rawFrMtrx{2}(pick,:) = fliplr(circshift(rawFrMtrx{1}(pick,:),[0,num_shift]));
    rawDeltaFrMtrx{2}(pick,:) = fliplr(circshift(rawDeltaFrMtrx{1}(pick,:),[0,num_shift]));
end

%% LIP neural activity accompanying decision termination (Fig. 3)
if fig_switch(1) || fig_switch(2)
    
    FH1 = figure(fig);clf;
    t_axis = 0:5000;
        % preallocation
    oneShapeKM = nan(5e5,100);
    woeVec = nan(5e5,2);
    woeFirstShapeKM = cell(1,8);
    
    order_list = [{'baseline'},{'1st'},{'2nd'},{'3rd'},{'4th'},{'5th'},...
                  {'6th'},{'7th'},{'8th'},{'9th'},{'10th'},{'11th'},{'12th'}];
    
    Tin_rew = logical(TM(:,1)==TM(:,7));

        % Randomize the trial order (not sorted by cells any more)
        % This step is necessary because if rows are sorted by cell identity, 
        % Some quantile would include cells with higher basline firing rates than
        % other quantiles, which would confound the analysis.
    KM_shuffled = KM(rand_trial_order,:);
    woeMtrx_shuffled = LLR{1}(rand_trial_order,:);
    rawWoeMtrx_shuffled = rawLLR{1}(rand_trial_order,:);
    cumWoeMtrx_shuffled = cumLLR{1}(rand_trial_order,:);
    num_accum_shuffled = num_accum(rand_trial_order);
    RT_shuffled = RT(rand_trial_order);
    
    firstShapeKM = KM_shuffled(:,1:5000)/FRgain;

    % Simple average
    uniqueWoe = [-9 -7 -5 -3 3 5 7 9];
    figure(fig);hold on
    
    num_bin = size(firstShapeKM,2);
    
    switch id
        case 1
            t_start = 100; %(ms)
            t_end = stim_int; %(ms)
        case 2
            t_start = 100; %(ms)
            t_end = stim_int; %(ms)
    end
    
    t_gap = 20;
    num_epoch = 18;
    
        % Quantiles are sorted by:
        % LLR (logLR) of the kth shape OR 'cumLLR' (cumulative logLR) of the 1st throgh kth shapes.
        % Use 'LLR' to plot delta FR (Fig. 3B), and 'cumLLR' to plot FR (Fig. 3A).    
    if fig_switch(1)
        sortBy = 'cumLLR';
    end
    if fig_switch(2)
        sortBy = 'LLR';
    end
        % preallocation
    beta_Mtrx = zeros(10,11);
    beta_se_Mtrx = zeros(10,11);
    beta_p_Mtrx = zeros(10,11);
    allEpochDeltaFR = cell(1,8);


    for ei = 1:num_epoch
        switch sortBy
            
            case 'cumLLR' % sort by cumulative logLR
                    % Divide trials into five quantiles
                group = 5; 
                    % Create color map
                map = colormap('jet');
                interval = floor((size(map,1)-1)/(group-1));
                color_map = map(1:interval:(1+interval*(group-1)),:);
                    % For the analysis of kth epoch, select trials in which N*>=k+1 (i.e. do not include N*=k) to 
                    % avoid contamination of saccadic preparation.
                pick = logical(isfinite(cumWoeMtrx_shuffled(:,ei)));
                pick = logical(pick & num_accum_shuffled>=ei+1);
                if sum(pick)==0
                    continue
                end
                    % Sort selected trials in order of cumulative logLR
                [sorted_cumWoe,sort_order] = sortrows(cumWoeMtrx_shuffled(pick,ei));
                sortedFirstShape = firstShapeKM(pick,:);
                sortedFirstShape = sortedFirstShape(sort_order,:);
                sorted_woe = woeMtrx_shuffled(pick,ei);
                sorted_woe = sorted_woe(sort_order);

                num_per_group = floor(length(sorted_cumWoe)/group);

                dfr_all = [];

                % Quantile analysis
                for gi = 1:group
                    pick_group = (num_per_group*(gi-1)+1):(num_per_group*gi);
                        % mean logLR for each quantile
                    woe_mean(gi) = mean(sorted_cumWoe(pick_group,end));
                    meanFirstShapeKM(gi,:) = mean(sortedFirstShape(pick_group,1:end-1),1);
                    sdFirstShapeKM(gi,:) = std(double(sortedFirstShape(pick_group,1:end-1)),0,1);
                    seFirstShapeKM(gi,:) = sdFirstShapeKM(gi,:)./sqrt(num_per_group);

                    figure(fig);hold on;
                    t_gap = 20;

                        % time window to plot FR
                    if ei==1
                        plot_pick = logical(t_axis<=t_axis(t_div+stim_int));
                    else
                        plot_pick = logical(t_axis>=t_axis(stim_int*(ei-1)+t_div+t_gap) & t_axis<=t_axis(stim_int*ei+t_div));
                    end

                        % time window to compute FR
                    t_pick = logical(t_axis>=t_axis(stim_int*(ei-1)+t_div+t_start) & t_axis<=t_axis(stim_int*(ei-1)+t_div+t_end));
                    fr_group{gi} = mean(sortedFirstShape(pick_group,t_pick),2);
                        % plot FR (mean±se)
                    fillTrace(t_axis(plot_pick),squeeze(meanFirstShapeKM(gi,plot_pick)),squeeze(seFirstShapeKM(gi,plot_pick)),color_map(gi,:));
                    hold on


                    if gi==group
                        analysis_range = t_axis(t_pick);
                            % Indicate analysis window by a grey bar on the abscissa
                        h = fillRect([analysis_range(1),0.1],[analysis_range(end),3],0.8*ones(1,3));
                        set(h,'EdgeColor','none')
                            % Divide the figure into subplots
                        plot_range = t_axis(plot_pick);
                        h = fillRect([plot_range(end),0],[plot_range(end)+t_gap,3],'w');
                        set(h,'LineWidth',3,'EdgeColor','w')
                        if ei>1
                            plot([plot_range(1) plot_range(1)],[0 70],'k-');
                        end
                    end
                end

                xlabel('Time from first shape onset (ms)','FontSize',24,'FontWeight','bold'); 
                ylabel('Firing rate (mean ± s.e.)','FontSize',24,'FontWeight','bold');
                xlim([0 1500]);
                ylim([0 70]);

                % compute for quintile plot %%%%%%%%%%%%%%%%%%%%%%%%%%%
                for gi = 1:group
                    info.fr_woe(ei,gi)  = woe_mean(gi)/10;
                    info.fr_mean(ei,gi) = mean(fr_group{gi});
                    info.fr_se(ei,gi) = stderr(fr_group{gi});
                end

                % plot FR for each unique cumWoe %%%%%%%%%%%%%%%%%%%%%%
                min_woe = min(sorted_cumWoe);
                max_woe = max(sorted_cumWoe);
                woe_center = min_woe:max_woe;
                woe_pick = logical(woe_center>(info.fr_woe(ei,1)*10-3) & woe_center<(info.fr_woe(ei,end)*10+3));
                info.fr_run_mean_xaxis{ei} = woe_center;

                clearvars fr_mean fr_sd fr_se
                for gi = 1:length(woe_center)
                    pick_group = logical(sorted_cumWoe==woe_center(gi));
                    if sum(pick_group)>=10
                        fr = mean(sortedFirstShape(pick_group,t_pick),2);
                        fr_mean(gi) = mean(fr);
                        fr_sd(gi) = std(fr);
                        fr_se(gi) = fr_sd(gi)/sqrt(sum(pick_group));
                    else
                        fr_mean(gi) = nan;
                        fr_sd(gi) = nan;
                        fr_se(gi) = nan;
                    end
                    info.fr_run_mean{ei}(gi) = fr_mean(gi);
                    info.fr_run_se{ei}(gi) = fr_se(gi);
                end

                if sum(isfinite(fr_mean))
                    % Fig. 3A, insets
                    figure(fig+1);
                    if ei<=10
                        subplot(2,5,ei);hold on;
                            % show the running mean of FR with grey shading
                        fillTrace(woe_center(woe_pick)/10,fr_mean(woe_pick),fr_se(woe_pick),0.3*[1 1 1]); hold on
                            % plot quintiles with color
                        for gi = 1:group
                            plot(woe_mean(gi)/10,mean(squeeze(meanFirstShapeKM(gi,t_pick))),'o','MarkerSize',12,'color',color_map(gi,:),'MarkerFaceColor',color_map(gi,:))
                        end
                    end
                    axis([-2.5 2.5 0 60]);
                    set(gca,'FontSize',30,'FontWeight','bold','Box','off','TickDir','out')
                    set(gca,'XTickMode','manual','XTick',[-2,0,2]);
                end
                
                %%%%% GLM fit to evaluate leverage of shapes on FR (reported in the paper) %%%%%
                [beta,dev,stats] = glmfit(woeMtrx_shuffled(pick,1:ei)/10,mean(firstShapeKM(pick,t_pick),2),'normal');

                % beta = beta/beta(1);
                beta_Mtrx(ei,1:ei+1) = beta;
                beta_se_Mtrx(ei,1:ei+1) = stats.se;
                beta_p_Mtrx(ei,1:ei+1) = stats.p;

                figure(fig+2)
                if ei<=10
                    subplot(2,5,ei); hold on
                    bar(0:ei,beta_Mtrx(ei,1:ei+1),0.7);
                    h = ploterr(0:ei,beta_Mtrx(ei,1:ei+1),[],beta_se_Mtrx(ei,1:ei+1),1,'k.','abshhy',0);
                    set(h(1),'MarkerSize',1)
                    xlim([-1 11])
                    ylim([0 40])
                    set(gca,'XTick',0:ei,'XTickLabel',0:ei,'ticklength',[0 0])
                    if ei ==1
                        fprintf('The leverage of the 1st shape on FR in\n') 
                    end
                    fprintf('epoch #%d: %.2f ± %.2f\t(p = %f)\n',ei,beta_Mtrx(ei,2),beta_se_Mtrx(ei,2),stats.p(2))
                end            
            
            
            case 'LLR'
                % create color table
                group = 8;
                map = colormap('jet');
                interval = floor((size(map,1)-1)/(group-1));
                color_map = map(1:interval:(1+interval*(group-1)),:);

                switch group
                    case 8 % sort by individual logLR
                            % time window to plot FR
                        t_plot_pick = logical(t_axis>=t_axis(stim_int*(ei-1)+t_div) & t_axis<=t_axis(stim_int*ei+t_div));
                            % time window to compute FR
                        t_pick = logical(t_axis>=t_axis(stim_int*(ei-1)+t_div+t_start) & t_axis<=t_axis(stim_int*(ei-1)+t_div+t_end));

                        if ei==1
                                % for the first epoch, the previous firing rate (FR0) estimated from 
                                % time between the first shape onset and the sensory delay (tau_s)
                            t_pre_pick = logical(t_axis<=t_axis(t_div));
                        else    % after the first epoch, define the previous epoch
                            t_pre_pick = logical(t_axis>=t_axis(stim_int*(ei-2)+t_div+t_start) & t_axis<=t_axis(stim_int*(ei-2)+t_div+t_end));
                        end

                        if isempty(firstShapeKM)
                            continue
                        elseif size(firstShapeKM,1)<2
                            deltaFR = (mean(double(firstShapeKM(:,t_pick)),2) - mean(double(firstShapeKM(:,t_pre_pick)),2));
                        else
                            deltaFR = (mean(double(firstShapeKM(:,t_pick)),2) - mean(double(firstShapeKM(:,t_pre_pick)),2));
                        end
                            % For the analysis of kth epoch, select trials in which N*>=k+1 (i.e. do not include N*=k) to 
                            % avoid contamination of saccadic preparation.
                        pick_epoch = logical(num_accum_shuffled>=ei+1);
                        pick_reg = logical(pick_epoch);
                        if sum(pick_reg)<1
                            continue
                        end

                        [beta dev stats] = glmfit(woeMtrx_shuffled(pick_reg,ei)/10,deltaFR(pick_reg));
                        grand_mean_deltaFR(ei) = mean(deltaFR(pick_epoch));
                        grand_se_deltaFR(ei) = stderr(deltaFR(pick_epoch));

                            % store values
                        offset(ei) = beta(1);
                        offset_se(ei) = stats.se(1);
                        slope(ei) = beta(2);
                        slope_se(ei) = stats.se(2);
                        p_values(ei) = stats.p(2);
                        num_used_dFR(ei) = sum(pick_epoch);

                        for ci = 1
                            for wi = 1:length(uniqueWoe)

                                meanDeltaFR(wi) = nan;
                                sdDeltaFR(wi) = nan;
                                seDeltaFR(wi) = nan;

                                pick_group = logical(pick_epoch & woeMtrx_shuffled(:,ei)==uniqueWoe(wi));
                                if sum(pick_group)<=1
                                    continue
                                end

                                woeFirstShapeKM{wi} = [woeFirstShapeKM{wi};firstShapeKM(pick_group,t_plot_pick)];

                                meanFirstShapeKM(ci,wi,:) = mean(double(firstShapeKM(pick_group,:)));
                                sdFirstShapeKM(ci,wi,:) = std(double(firstShapeKM(pick_group,:)));
                                seFirstShapeKM(ci,wi,:) = sdFirstShapeKM(ci,wi,:)/sqrt(sum(pick_group));
                                meanRT(ci,wi) = mean(RT_shuffled(pick_group));

                                meanDeltaFR(wi) = mean(deltaFR(pick_group));
                                sdDeltaFR(wi) = std(deltaFR(pick_group));
                                seDeltaFR(wi) = sdDeltaFR(wi)/sqrt(sum(pick_group));
                                dfr_axis = -40:5:40;
                                temp = histc(deltaFR(pick_group),dfr_axis);

                                if wi<5
                                    M(:,wi) = temp/sum(temp);
                                else
                                    M(:,wi+2) = temp/sum(temp);
                                end

                                allEpochDeltaFR{wi} = [allEpochDeltaFR{wi};deltaFR(pick_group)];

                                figure(fig);hold on;
                                t_plot = t_axis(t_plot_pick);
                                meanFirstShapeKM_plot = squeeze(meanFirstShapeKM(ci,wi,t_plot_pick));
                                seFirstShapeKM_plot = squeeze(seFirstShapeKM(ci,wi,t_plot_pick));
                                fillTrace(t_plot(1:end-t_gap),meanFirstShapeKM_plot(1:end-t_gap),seFirstShapeKM_plot(1:end-t_gap),color_map(wi,:));
                                hold on
                                if ei>1
                                    plot([t_plot(1) t_plot(1)],[0 70],'k-');
                                end
                                xlabel('Time from first shape onset (ms)','FontSize',24,'FontWeight','bold'); 
                                ylabel('Firing rate (mean ± s.e.)','FontSize',24,'FontWeight','bold');
                                xlim([0 1500]);
                                ylim([0 70]);
                            end
                        end

                        figure(fig)
                        xlim([0 t_axis(stim_int*(ei+1)+t_div)])
                        switch id
                            case 1
                                ylim([0 50])
                            case 2
                                ylim([0 70])
                        end
                            % Plot FR vs cumulative logLR (Fig. 3A)
                        figure(fig+1);
                        if ei<=10
                            subplot(2,5,ei);hold on;

                            for wi = 1:length(uniqueWoe)
                                h = ploterr(uniqueWoe(wi)/10,meanDeltaFR(wi),[],seDeltaFR(wi),0.5,'ko','abshhy',0);
                                set(h(1),'MarkerSize',12,'color','k','MarkerFaceColor','k')
                            end
                            xlim([-1.3 1.3])
                            switch id
                                case 1
                                    ylim([-30 30]);
                                case 2
                                    ylim([-50 50]);
                            end
                            y_range = diff(get(gca,'ylim'));
                            h1 = fill([-1.3,-1.3,-1.15],[grand_mean_deltaFR(ei)+y_range*0.035,grand_mean_deltaFR(ei)-y_range*0.035,grand_mean_deltaFR(ei)],'r');
                            set(h1,'EdgeColor','none')

                            title(sprintf('%s to %s',order_list{ei},order_list{ei+1}),'FontSize',16,'FontWeight','bold')
                            y_lim = get(gca,'Ylim');
                            maxY = y_lim(2);

                            if 1
                                text(-1.1,maxY*0.9,sprintf('n = %d',num_used_dFR(ei)),'FontSize',16,'FontWeight','bold');
                            end

                        end
                            % Format figures
                        figure(fig)
                        set(gca,'FontSize',20,'FontWeight','bold','Box','off','TickDir','out')
                        set(gca,'XTickMode','manual','XTick',0:stim_int:stim_int*6);

                        figure(fig+1)
                        set(gca,'FontSize',20,'FontWeight','bold','Box','off','TickDir','out')
                        set(gca,'XTickMode','manual','XTick',[-1,0,1]);

                    case 4 % sort by individual logLR -- used to compute the sensory delay (tau_s) (see Exp. Proc.)

                        if ei>1
                            continue
                        end

                        for ri = 1
                            for wi = 1:length(uniqueWoe)/2
                                w_pick{wi} = logical((rawWoeMtrx_shuffled(:,ei)==uniqueWoe(2*wi-1) | rawWoeMtrx_shuffled(:,wi)==uniqueWoe(2*wi)));
                                meanFirstShapeKM(ri,wi,:) = mean(double(firstShapeKM(w_pick{wi},1:end)));
                                sdFirstShapeKM(ri,wi,:) = std(double(firstShapeKM(w_pick{wi},1:end)));
                                seFirstShapeKM(ri,wi,:) = sdFirstShapeKM(ri,wi,:)/sqrt(sum(w_pick{wi}));
                                meanRT(ri,wi) = mean(RT_shuffled(w_pick{wi}));

                                t_pick = logical(t_axis<=t_axis(500));
                                meanFirstShapeKMtemp = squeeze(meanFirstShapeKM(ri,wi,t_pick));

                                xlabel('Time from 1st shape onset (ms)','FontSize',18,'FontWeight','bold'); 
                                ylabel('Firing rate (mean±sem)','FontSize',18,'FontWeight','bold');

                                xlim([min(t_axis(t_pick)) max(t_axis(t_pick))])

                                figure(fig)
                                hold on;
                                p=fillTrace(t_axis(t_pick)+2-ei,meanFirstShapeKMtemp,1*squeeze(seFirstShapeKM(ri,wi,t_pick)),color_map(wi,:));
                                xlabel('Time from 1st shape onset','FontSize',30,'FontWeight','bold'); 
                                ylabel('FR ± SD','FontSize',30,'FontWeight','bold');
                                xlim([min(t_axis(t_pick)) max(t_axis(t_pick))])
                                set(gca,'FontSize',30,'FontWeight','bold','Box','off','TickDir','out')
                                set(gca,'XTick',0:50:400,'XTickLabel',[{'0'},{''},{'100'},{''},{'200'},{''},{'300'},{''},{'400'}])
                            end
                        end
                end
        end
    end
    
        % Plot delta FR (Fig. 3B)
    if strcmp(sortBy,'LLR') 
        if group==8
            allDeltaFR = [];
            allWOE = [];
            for gi = 1:length(uniqueWoe)
                allEpochFRmean(gi) = nanmean(allEpochDeltaFR{gi});
                allEpochFRsd(gi) = nanstd(allEpochDeltaFR{gi});
                allEpochFRse(gi) = allEpochFRsd(gi)/sqrt(length(allEpochDeltaFR{gi}));
                allEpochFRnum(gi) = sum(isfinite(allEpochDeltaFR{gi}));
                allDeltaFR = [allDeltaFR;allEpochDeltaFR{gi}];
                allWOE = [allWOE;uniqueWoe(gi)*ones(size(allEpochDeltaFR{gi}))];
            end
            allEpochFRmeanZero = allEpochFRmean-mean(allDeltaFR); % used as 'delta r' for modeling (Fig. 5B)
            figure(fig+2);hold on
            h = ploterr(uniqueWoe/10,allEpochFRmean,[],allEpochFRse,1,'ko','abshhy',0);
            set(h(1),'MarkerSize',8,'MarkerFaceColor','k');
            xlim([-1.1 1.1])
            ylim([-30 30])
            xlabel('Evidence for Tin (logLR)','FontSize',18,'FontWeight','bold')
            ylabel('Change in firing rate (sp/s)','FontSize',18,'FontWeight','bold')
            set(gca,'FontSize',24,'FontWeight','bold','Box','off','TickDir','out')
            set(gca,'YTick',-30:10:30);
            switch id
                case 1
                    ylim([-5 15])
                    set(gca,'YTick',-5:5:15);
                case 2
                    ylim([-30 30])
                    set(gca,'YTick',-30:10:30);
            end
            set(gcf,'Color','w','position',[1500 800 400 400])
        end
    end
        % Format figures
    if group~=4
        figure(fig)
        xlim([0 1450])
        ylim([0 70])
        set(gcf,'position',[100 100 1400 600],'Color','w')
        set(gca,'FontSize',30,'FontWeight','bold','Box','off','TickDir','out')
        set(gca,'XTickMode','manual','XTick',0:250:250*10);
    
        figure(fig+1)
        set(gcf,'position',[100 800 1400 600],'Color','w')
    end
    
    if strcmp(sortBy,'cumLLR')
        figure(fig+2)
        set(gcf,'position',[100 700 1150 450],'Color','w')
    end

    fig = fig+3;
end

%% LIP neural activity accompanying decision termination (Fig. 4A)
if fig_switch(3)
    
    num_fig = 5;
    
    group = 5;
    map = colormap('jet');
    interval = floor((size(map,1)-1)/(group-1));
    color_map = map(1:interval:(1+interval*(group-1)),:);

    % Extract the peri-saccadic FR [-500ms 150ms]
    t_pre_sac = -500;
    t_post_sac = 150;
    t_axis = t_pre_sac:t_post_sac;

    t_center_ind = 1:length(t_axis);
    t_center = t_axis(t_center_ind);
        
        
    sacKM = nan(size(KM,1),t_post_sac-t_pre_sac+1);
    for ti = 1:size(KM,1)
        RT_ind = round(RT(ti))+1;
        t_min_ind = RT_ind+t_pre_sac;
        t_max_ind = RT_ind+t_post_sac;
        if t_min_ind<1
            sacKM(ti,-t_min_ind+2:end) = KM(ti,1:t_max_ind);
        else
            sacKM(ti,:) = KM(ti,t_min_ind:t_max_ind);
        end
    end

    lastStim2Sac = TM(:,4);

        % Randomize the trial order (not sorted by cells any more)
        % KM stores FR multiplied by FRgain (=100) in integer, so divide them by FR gain to obtain the original FR values
    sacKM_shuffled = sacKM(rand_trial_order,:)/FRgain;
    T_in_out_shuffled = T_in_out(rand_trial_order);
    correct_shuffled = correct(rand_trial_order');
    rawCumLLR_shuffled = rawCumLLR{2}(rand_trial_order,:);
    lastStim2Sac_shuffled = lastStim2Sac(rand_trial_order);

    extra_delay = 100; %(ms) taking into account a rise time of FR after the shape onset.

    for di = 1:length(extra_delay)

            % create a matrix of the total logLR at each point in time.
        cumWoeKM = nan(size(KM,1),length(t_axis));
            % time when logLR at saccade is reflected in FR
        sac_ind = -t_pre_sac + info.delay + extra_delay(di) + 1; 
        for ti = 1:size(KM,1)
            t_lastStim_ind = sac_ind-round(lastStim2Sac_shuffled(ti));
            if t_lastStim_ind<1
                continue
            end
            cumWoeKM(ti,t_lastStim_ind:end) = rawCumLLR_shuffled(ti,1);
            for si = 1:5
                if t_lastStim_ind-stim_int*si>0
                    t_pick = (t_lastStim_ind-stim_int*si):(t_lastStim_ind-stim_int*(si-1)-1);
                    cumWoeKM(ti,t_pick) = rawCumLLR_shuffled(ti,si+1);
                else
                    t_pick = 1:t_lastStim_ind-stim_int*(si-1)-1;
                    cumWoeKM(ti,t_pick) = rawCumLLR_shuffled(ti,si+1);
                    break
                end
            end
        end

        bin_size = 10;
        bin_shift = 10;
        T_label = t_pre_sac:t_post_sac;
        T_edge = t_pre_sac:bin_shift:t_post_sac;

        figure(fig);clf;
        set(gcf,'position',[100 900 750 350],'Color','w');
            
            % Set i appropriately to plot Tin or Tout trials
            % 1: Tin, 2: Tout
        for i = 2:-1:1
            pick = logical(T_in_out_shuffled==i);
            
            fr_mean_group{i} = nan(group,length(t_center));
            fr_se_group{i} = nan(group,length(t_center));

            for ti = 1:length(t_center);
                t_pick = ti;

                cumWoe_pick = mean(cumWoeKM(pick,t_pick),2);
                fr_pick = mean(sacKM_shuffled(pick,t_pick),2);
                correct_pick = correct_shuffled(pick);

                pick_finite = isfinite(cumWoe_pick);

                if sum(pick_finite)<=1
                    continue
                end

                cumWoe_finite = cumWoe_pick(pick_finite);
                fr_finite = fr_pick(pick_finite);
                correct_finite = correct_pick(pick_finite);
                bigS = nan(sum(pick_finite),1);
                
                % Here, each value of cumulative logLR might induce a different variance in FR.
                % So we will perform a weighted regression analysis in which weights are given by 
                % inverses of the variance at each value of cumulative logLR.
                clearvars fr_mean fr_se
                uniqueCumWoe = unique(cumWoe_finite);
                for wi = 1:length(uniqueCumWoe)
                    pick_woe = logical(cumWoe_finite==uniqueCumWoe(wi));
                    if sum(pick_woe)>=10
                        fr_mean(wi) = mean(fr_finite(pick_woe));
                        fr_se(wi) = stderr(fr_finite(pick_woe));
                        bigS(pick_woe) = std(fr_finite(pick_woe));
                    else
                        fr_mean(wi) = nan;
                        fr_se(wi) = nan;
                        bigS(pick_woe) = nan;
                    end
                end

                %%%%%%%%%%%%%%%%% weighted regresssion analysis %%%%%%%%%%%%%%%%
                [beta,dev,stats] = glmfit(cumWoe_finite/10,fr_finite,'normal','weight',bigS.^-2);

                slope_v(ti) = beta(2);
                slope_se_v(ti) = stats.se(2);
                slope_p_v(ti) = stats.p(2);

                [cumWoe_sorted,sort_order] = sortrows(cumWoe_finite);
                fr_sorted = fr_finite(sort_order);

                num_per_group = floor(sum(pick_finite)/group);
                for gi = 1:group
                    pick_group = (num_per_group*(gi-1)+1):(num_per_group*gi);
                    fr_group{i,gi,ti} = fr_sorted(pick_group);
                    fr_mean_group{i}(gi,ti) = nanmean(fr_sorted(pick_group));
                    fr_se_group{i}(gi,ti) = nanstd(fr_sorted(pick_group))/sqrt(num_per_group);
                end

                fr_raw{i,ti} = fr_finite/FRgain;
                fr_raw_correct{i,ti} = fr_finite(correct_finite);
                fr_raw_error{i,ti}   = fr_finite(~correct_finite);

            end
            for gi = 1:group
                plot_color = 1-(1-color_map(gi,:))/(1+0.5*(i-1)); % Tout is plotted with a tint color.
                fillTrace(t_center(3:end),fr_mean_group{i}(gi,3:end),1*fr_se_group{i}(gi,3:end),plot_color);
            end
        end

        hold on
        plot([0 0],[0 100],'k--');
        ylim([0 80])
        xlim([t_pre_sac,t_post_sac])
        set(gca,'XTick',-500:100:100,'YTick',0:10:70);
        xlabel('Time from saccade (ms)','FontSize',18,'FontWeight','bold')
        ylabel('Firing rate (sp/s)','FontSize',18,'FontWeight','bold')
        
        
            % plot p-value for FR vs cumWoe slope
            % when does FR converge (= tau_m)? (reported in the paper)
        figure(fig+1);hold on
        plot(t_center,slope_p_v,'bo-')
        plot(t_center,0.05*ones(1,length(t_center)),'k--')
        xlim([-500 150])
        xlabel('Time from saccade (ms)','FontSize',18,'FontWeight','bold')
        ylabel('p-val','FontSize',24,'FontWeight','bold')
        set(gcf,'position',[100 300 300 200],'Color','w');
            
            % Fig. 4A insets
            % Regression slope for [FR vs cumulative logLR] as a function of time
        figure(fig+2); hold on
        fillTrace(t_center,slope_v,slope_se_v,0*[1,1,1],0.3);hold on
        plot([0 0],[-10 30],'k--')
        plot(t_center,zeros(1,length(t_center)),'k--')
        xlim([-500 150])
        switch id
            case 1
                ylim([-3 4])
            case 2
                ylim([-5 20])
        end
        xlabel('Time from saccade (ms)','FontSize',18,'FontWeight','bold')
        ylabel('Regression slope','FontSize',18,'FontWeight','bold')
        set(gca,'XTick',-500:100:100,'XTickLabel',[{''},{'-400'},{''},{'-200'},{''},{'0'},{''}])
        set(gcf,'position',[100 600 300 200],'Color','w');
        
        % Presaccadic FR reaches the same level for correct and error Tin trials? (reported in the paper)
        t_ind = find(t_center>=-50,1,'first');
        [h,p,ci,stats] = ttest2(fr_raw_correct{1,t_ind},fr_raw_error{1,t_ind});
        fprintf('H0: Presaccadic FR are the same for correct and error Tin trials\n')
        fprintf('p = %f\n\n',p)
    end
        
    fig = fig+num_fig;

end

%% Change in LIP neural activity accompanying decision termination (Fig. 4B)
if fig_switch(4)

    lastStim2Sac = TM(:,4);
    
        % lastStim2Sac is the time from N*th shape to saccade initiation 
    while sum(lastStim2Sac<lastAccumShape2Sac)
        pick = logical(lastStim2Sac<lastAccumShape2Sac);
        lastStim2Sac(pick) = lastStim2Sac(pick)+stim_int;
    end

        % Randomize the trial order (not sorted by cells any more)
        % This step is necessary because if rows are sorted by cell identity, 
        % Some quantile would include cells with higher basline firing rates than
        % other quantiles, which would confound the analysis.
    KM_shuffled = KM(rand_trial_order,:);
    LLR_shuffled = LLR{2}(rand_trial_order,:);
    num_accum_shuffled = num_accum(rand_trial_order);
    T_in_out_shuffled = T_in_out(rand_trial_order);
    lastStim2Sac_shuffled = lastStim2Sac(rand_trial_order);
    
    woe = [-9 -7 -5 -3 3 5 7 9];
    woe2analyze = 1:8;

    t_offset = 500;
    group = 8;
    
    color_group = group;
    map = colormap('jet');
    interval = floor((size(map,1)-1)/(group-1));
    color_map = map(1:interval:(1+interval*(color_group-1)),:);
    
    % Do not touch this for plot %%%%%%%%%
    t_start = 100;
    t_end = stim_int;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % indicatde the time window for saccade (mean ± s.d.)
                
    plot_start = t_div;
    plot_end = t_div+500;

    t_oneShape = plot_start:plot_end;

        % preallocation 
    oneShapeKM     = nan(size(KM,1),plot_end-plot_start+1);
    oneShapeKM_pre = nan(size(KM,1),plot_end-plot_start+1);

    for i = 1:size(KM_shuffled,1)
        if num_accum_shuffled(i) <= 0; % don't change
            continue
        end
        t_label = stim_int*(num_accum_shuffled(i)-1)+1+plot_start:stim_int*(num_accum_shuffled(i)-1)+1+plot_end;
        % t_label = 1:stim_int*(num_accum_shuffled(i)-ei)+plot_end;
        oneShapeKM(i,:) = KM_shuffled(i,t_label);

        if num_accum_shuffled(i) == 1 % include N*=1 trials
            % for trials with N*=1, shift the data by t_start so that 
            % the interval between the 1st shape onset and sensory delay (tau_s) 
            % will be selected for later analyses
            t_label = 1:t_div;
            oneShapeKM_pre(i,t_start+1:t_start+t_div) = KM_shuffled(i,t_label);
        else
            t_label = stim_int*(num_accum_shuffled(i)-2)+1+plot_start:stim_int*(num_accum_shuffled(i)-2)+1+plot_end;
            oneShapeKM_pre(i,:) = KM_shuffled(i,t_label);
        end
    end

        % KM stores FR multiplied by FRgain (=100) in integer, so divide them by FR gain to obtain the original FR values
    oneShapeKM = oneShapeKM/FRgain;
    oneShapeKM_pre = oneShapeKM_pre/FRgain;

    for i = 1:2 % Tin or Tout
        clearvars epochGroupFR meanFirstShapeDeltaKM sdFirstShapeDeltaKM seFirstShapeDeltaKM quantile_var
        pick_in_out = logical(T_in_out_shuffled==i);
        pick_sign = logical((-1)^(i-1)*LLR_shuffled(:,1)>0); % choose positive logLR for Tin and negative for Tout
        pick_opposite_sign = logical((-1)^(i-1)*LLR_shuffled(:,1)<0); % choose ngative logLR for Tin and positive for Tout

        t_temp = plot_start:plot_end;
            % timewindow to compute FR
        t_pick = logical(t_temp>=(t_div+t_start) & t_temp<=(t_div+t_end));

        switch i
            case 1
                woe2analyze = 1:8;
                woe2regress = 5:8;
            case 2
                woe2analyze = 1:8;
                woe2regress = 1:4;
        end
        
            % delta FR conditioned on logLR associated with N*
        for gi = woe2analyze
            pick_woe = logical(LLR_shuffled(:,1)==woe(gi));
            pick = logical(pick_in_out & pick_woe);
            fr_picked = oneShapeKM(pick,:);
            fr_picked_pre = oneShapeKM_pre(pick,:);
            lastStim2Sac_picked = lastStim2Sac_shuffled(pick);

            epochGroupDeltaFR = nan(size(lastStim2Sac_picked)); % pre-allocation
            for ti = 1:size(lastStim2Sac_picked,1)
                t_pick_Nstar = logical(t_temp<=lastStim2Sac_picked(ti));
                if 1
                    epochGroupDeltaFR(ti) = nanmean(double(fr_picked(ti,t_pick_Nstar)),2) - nanmean(double(fr_picked_pre(ti,t_pick)),2);
                else
                    epochGroupDeltaFR(ti) = nanmean(double(fr_picked(ti,t_pick)),2) - nanmean(double(fr_picked_pre(ti,t_pick)),2);
                end
            end
            epochGroupDeltaFR_mean(i,gi) = nanmean(epochGroupDeltaFR);
            epochGroupDeltaFR_se(i,gi) = nanse(epochGroupDeltaFR);
        end
        
            % Does delta FR associated with N* shapes reflect their logLR?
        epochDeltaFR_all = nan(size(lastStim2Sac_shuffled)); % pre-allocation
        for ti = 1:size(lastStim2Sac_shuffled,1)
            t_pick_Nstar = logical(t_temp<=lastStim2Sac_shuffled(ti));
            epochDeltaFR_all(ti) = nanmean(double(oneShapeKM(ti,t_pick_Nstar)),2)... 
                                 - nanmean(double(oneShapeKM_pre(ti,t_pick)),2);
        end
        pick = (pick_in_out & pick_sign);
        epochDeltaFR = epochDeltaFR_all(pick);

            % Here, each value of logLR might induce a different variance in delta FR.
            % So we will perform a weighted regression analysis in which weights are given by 
            % inverses of the variance at each value of logLR.
        LLR_shuffled_picked = LLR_shuffled(pick,1);
        bigS = nan(size(LLR_shuffled_picked));            
        for wi = 1:length(woe(woe2regress))
            pick_LLR = logical(LLR_shuffled_picked==woe(woe2regress(wi)) & isfinite(epochDeltaFR));
            bigS(pick_LLR) = nanstd(epochDeltaFR(pick_LLR));
        end

            % GLM fit to evaluate leverage of shapes on delta FR (weighted regression) %%%%%
        [betaGLM,devGLM,stats] = glmfit(LLR_shuffled_picked/10,epochDeltaFR,'normal','weights',bigS.^-2);
        statsGLM = stats;
        if i==1
                % positive N* shapes have differential effects on delta FR (reporeted in the paper) 
            betaGLM_p = stats.p;
            fprintf('slope for delta FR vs logLR: %.1f ± %.1f sp/[s*logLR] (p = %f)\n\n',betaGLM(2),stats.se(2),betaGLM_p(2));
                % even shapes associated with negative logLR induce positive change in FR? (reported in the paper)
            epochDeltaFR_opposite = epochDeltaFR_all(pick_in_out & pick_opposite_sign);  
            [h,p,ci,stats] = ttest(epochDeltaFR_opposite);
            fprintf('delta FR for negative shapes: %.1f ± %.1f sp/s (p = %f)\n\n',nanmean(epochDeltaFR_opposite),nanse(epochDeltaFR_opposite),p);
        end
            % Plot mean delta FR associated with N* (Fig. 4B)
        figure(fig+i-1); hold on
        color_ind = 1;
        for wi = woe2analyze
            h = ploterr(woe(wi)/10,epochGroupDeltaFR_mean(i,wi),[],epochGroupDeltaFR_se(i,wi),1,'ko','abshhy',0);
            set(h(1),'MarkerSize',8,'color','k','MarkerFaceColor','k')
            color_ind = color_ind+1;
        end
        plot([woe(woe2regress(1)),woe(woe2regress(end))]/10,statsGLM.beta(1)+statsGLM.beta(2)*[woe(woe2regress(1)),woe(woe2regress(end))]/10,'k--')
            % Format figures
        switch id
            case 1
                switch i
                    case 1
                        ylim([0 20])
                    case 2
                        ylim([-10 10])
                end
            case 2
                ylim([-10 10])
        end
        figure(fig+i-1)
        set(gcf,'position',[100+450*(i-1) 100 400 400],'Color','w')

    end
    fig = fig+2;
end