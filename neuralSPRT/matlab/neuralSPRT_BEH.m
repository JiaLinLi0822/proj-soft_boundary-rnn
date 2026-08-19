function info = neuralSPRT_BEH(info,id,fig_switch)

% fig_switch:
% 1: Fig. 2A -- RT histogram
% 2: Fig. 2B -- Effect of a single shape presented at a specific timing
% 3: Fig. 2C -- N* histogram
% 4: Fig. 2D -- psychometric function
% 5: Fig. 2E -- Cumulative evidence at the end of decision (also Fig. S1)
% 6: Fig. 2F -- Subjective weight

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

%%
% lastAccumShape2Sac spcifies the non-decision time estimated by behavior.
% That is, the last one or two shapes presented within this time window
% prior to saccade did not have leverage on behavior (see Fig. 2B right).
% Set "lastAccumShape2Sac" to 0 (ms) to include all the observed shapes (as 
% opposed to accumulated shapes) in order to plot Fig. 2B


switch id
    case 1  % monkey E
        lastAccumShape2Sac = 270; %(ms) Non-decision time estimated by behavior (see Fig. 2B right)
        stim_int = 250;
    case 2  % monkey J
        lastAccumShape2Sac = 180; %(ms) Non-decision time estimated by behavior (see Fig. 2B right)
        stim_int = 270;
end
if fig_switch(2)
    % Set "lastAccumShape2Sac" to 0 (ms) to include all the observed shapes 
    % (as opposed to accumulated shapes) in order to plot Fig. 2B
    lastAccumShape2Sac = 0; %(ms)
end

fig = info.fig;
  
%% Pre-processing the data
    
TM = info.TM;

total_shape = 8; % the total number of unique shapes (see Fig. 1B).
shape_offset = 2;

    % Remove trials with trump shapes (monkey J)
shapeMtrx = TM(:,11:30)+1; % shapes are labeled as #1-10 in shapeMtrx
pick = logical(sum(shapeMtrx==1|shapeMtrx==2,2));
num_trump_trial = sum(pick);
shapeMtrx = shapeMtrx(~pick,:);
TM = TM(~pick,:);

    % true assigned weights (logLR) multiplied by 10 to make them integers.
trueDeciWOE = [9999 -9999 9 -9 7 -7 5 -5 3 -3 nan]; 
    % location of presented shapes. 1: upper left, 2: upper right, 3:lower left, 4: lower right
if size(TM,2)>70
    locMtrx = TM(:,71:90);
end
    % number of presented shapes before saccade initiation (N). (N>=N*) 
num_shown = sum(isfinite(shapeMtrx),2);
    % reward-assigned target (1:left target, 2:right target)
rew_targ = TM(:,1);
if id==2
    % For monkey J only -- reward-assigned target (1:Green target, 2:Red target)
    rew_color = (rew_targ~=TM(:,9))+1;
end
    % monkey's choice
    % Target A/B are left/right for monkey E, and red/green for monkey J.
    % negative evidence favors choice 1 = Target B (E:Right,  J:Green)
    % positive evidence favors choice 2 = Target A (E:Left, J:Red)
switch id
    case 1 % Eli
        % IMPORTANT: sign was changed for a simpler sign convention
        choice = 3-TM(:,2); % 1: right choice, 2: left choice
        rew_targ = 3-rew_targ; % 1: right target, 2: left target
    case 2 % Joey
        choice = TM(:,10); % 1: Green, 2: Red
end
    % reaction time
RT = TM(:,3); % (ms)

    % time from the onset of last presented shape (Nth) to saccade initiation.
lastStim2Sac = TM(:,4); % (ms)

    % decision outcome -- 0: error, 1:correct
correct = TM(:,1)==TM(:,2);

    % create a matrix (rawLLR) contatining assigned weights for presented shapes
rawShapeMtrx = shapeMtrx;
rawShapeMtrx(isnan(rawShapeMtrx))=11;
rawLLR{1} = trueDeciWOE(rawShapeMtrx);

    % compute N* (num_accum) for each trial based on the non-decision time 
    % specified above (lastAccumShape2Sac)
fprintf('compute N*\n\n');
num_accum = num_shown;
t_cutoff = lastAccumShape2Sac*ones(length(num_shown),1);
t_cutoff = t_cutoff-TM(:,4);
pick = t_cutoff>0;
num_accum(pick) = num_accum(pick)-1;
while any(t_cutoff>0)
    t_cutoff = t_cutoff-stim_int;
    pick = t_cutoff>0;
    num_accum(pick) = num_accum(pick)-1;
end

% num_accum = ceil((RT-lastAccumShape2Sac)/stim_int);

    % replace the assigned weights with NaNs for shapes shown after the N*th 
for i = 1:size(TM,1)
    if num_accum(i)<0;
        num_accum(i)=0;
    end
    shapeMtrx(i,num_accum(i)+1:num_shown(i)) = nan;
    locMtrx(i,num_accum(i)+1:num_shown(i)) = nan;
end
    % Now shapeMtrx contains only the 1st through N*th shapes.

shapeMtrx(isnan(shapeMtrx))=11;
LLR{1} = trueDeciWOE(shapeMtrx); % assigned weights for the 1st through N*th shapes

%%%%%%%%%%%% IMPORTANT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% changing the sign of weight for monkey E (id=1)
% such that positive evidence favors Left (Tin)
% to make the sign convention simpler in the paper.
%
    if id==1
        rawLLR{1} = -rawLLR{1};
        LLR{1} = -LLR{1};
    end
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



cumLLR{1} = cumsum(LLR{1},2);
rawCumLLR{1} = cumsum(rawLLR{1},2);

% For later analyses, reverse the element order in each row.  
LLR{2} = nan(size(LLR{1}));
cumLLR{2} = nan(size(cumLLR{1}));
rawLLR{2} = nan(size(rawLLR{1}));
rawCumLLR{2} = nan(size(rawCumLLR{1}));    
for ci = 1:20
    % dealing with accumulated shapes
    pick = logical(num_accum==ci); 
    num_shift = 20-ci;
    LLR{2}(pick,:) = fliplr(circshift(LLR{1}(pick,:),[0,num_shift]));
    cumLLR{2}(pick,:) = fliplr(circshift(cumLLR{1}(pick,:),[0,num_shift]));

    % dealing with shown shapes
    pick = logical(num_shown==ci);
    rawLLR{2}(pick,:) = fliplr(circshift(rawLLR{1}(pick,:),[0,num_shift]));
    rawCumLLR{2}(pick,:) = fliplr(circshift(rawCumLLR{1}(pick,:),[0,num_shift]));
end

    % total cumulative logLR for each trial
totalLLR = nansum(LLR{1},2);

%% RT histogram (Fig. 2A)
if fig_switch(1)
    
    t_bin = 0:10:5000;

    rt_hist = histc(RT,0:5000);
    rt_cum_hist = cumsum(rt_hist);
    rt_hist_10ms = diff([0;rt_cum_hist(t_bin+1)]);
    p_rt_hist = rt_hist./sum(rt_hist);
    cum_p_rt_hist = cumsum(p_rt_hist);
    p_rt_hist_10ms = diff([0;cum_p_rt_hist(t_bin+1)]);

    rt_hist_correct = histc(RT(correct==1),0:5000);
    rt_cum_hist_correct = cumsum(rt_hist_correct);
    rt_hist_10ms_correct = diff([0;rt_cum_hist_correct(t_bin+1)]);
    p_rt_hist_correct = rt_hist_correct./sum(rt_hist);
    cum_p_rt_hist_correct = cumsum(p_rt_hist_correct);
    p_rt_hist_10ms_correct = diff([0;cum_p_rt_hist_correct(t_bin+1)]);

    rt_hist_error = histc(RT(correct==0),0:5000);
    rt_cum_hist_error = cumsum(rt_hist_error);
    rt_hist_10ms_error = diff([0;rt_cum_hist_error(t_bin+1)]);
    p_rt_hist_error = rt_hist_error./sum(rt_hist);
    cum_p_rt_hist_error = cumsum(p_rt_hist_error);
    p_rt_hist_10ms_error = diff([0;cum_p_rt_hist_error(t_bin+1)]);

    figure(fig);hold on
    bar(t_bin/1e3,rt_hist_10ms,1,'k')
    xlim([0 3])
    ylim([0 600])
    set(gca,'FontSize',24,'FontWeight','bold','Box','OFF','TickDir','out');
    set(gca,'XTick',0:0.25:3,'XTickLabel',[{'0'},{''},{''},{''},{'1'},{''},{''},{''},{'2'},{''},{''},{''},{'3'}]);
    set(gca,'YTick',0:200:600)
    set(gcf,'position',[200 500 400 200])
    
    %%%%%%% save info %%%%%%
    
    info.rt_hist = rt_hist;
    info.rt_hist_correct = rt_hist_correct;
    info.rt_hist_error = rt_hist_error;
    
    info.p_rt_hist_10ms = p_rt_hist_10ms;
    info.p_rt_hist_10ms_correct = p_rt_hist_10ms_correct;
    info.p_rt_hist_10ms_error = p_rt_hist_10ms_error;
    
    info.rt_hist_10ms = rt_hist_10ms;
    info.rt_hist_10ms_correct = rt_hist_10ms_correct;
    info.rt_hist_10ms_error = rt_hist_10ms_error;
    
    fig = fig + 1;
end

%% Logistic regression by GLM fit (Fig. 2B)
if fig_switch(2)
    
    % *********** Forward Analysis ***********
    
    LLR_forward = rawLLR{1}/10;
    num_epoch = 10;
    
    combine_choice_flag = 1;
    offset_flag = 0;
    switch offset_flag
        case 0
            const = 'off';
        case 1
            const = 'on';
    end
    
    for ei = 1:num_epoch
        for ri = 1:2;
            pick = logical(isfinite(LLR_forward(:,ei)));
            LLR_picked = LLR_forward(pick,ei);
            res_col = setdiff(1:10,ei);
            resLLR_picked = nansum(LLR_forward(pick,res_col),2);
            choice_picked = choice(pick);
            [logitCoef,dev,stats] = glmfit([LLR_picked, resLLR_picked],choice_picked-1,'binomial','link','logit','constant',const);
                % Convert the log base from exp to 10
            logitCoef = logitCoef/log(10);
            stats.se = stats.se/log(10);
            
            glm_w(ei,:) = logitCoef;
            glm_w_se(ei,:) = stats.se;
            glm_w_ci(ei,:) = stats.se * norminv(0.975,0,1);
            glm_w_lo(ei,:) = logitCoef - stats.se * norminv(0.975,0,1);
            glm_w_hi(ei,:) = logitCoef + stats.se * norminv(0.975,0,1);
            glm_p(ei,:) = stats.p;  
        end
    end
    
    %%%%%% save info %%%%%%
    info.glm_w_init    = glm_w;
    info.glm_w_se_init = glm_w_se;
    info.glm_w_ci_init = glm_w_ci;
    info.glm_w_lo_init = glm_w_lo;
    info.glm_w_hi_init = glm_w_hi;
    info.glm_p_end     = glm_p;
    
    figure(fig);
    subplot('position',[0.1,0.1,0.8*6/11,0.85]);hold on
    t_onset = ((1:num_epoch)-1)*stim_int;
    for bi = 1:5
        fillTrace([t_onset(bi)-10,t_onset(bi)+10],squeeze(glm_w(bi,2))*[1,1],squeeze(glm_w_se(bi,2))*[1,1],0.2*[1,1,1]);
        hold on
        plot([t_onset(bi),t_onset(bi+1)],squeeze(glm_w(bi,2))*[1,1],'k--')
        plot([t_onset(bi+1),t_onset(bi+1)],squeeze(glm_w([bi,bi+1],2)),'k--')
    end
    fprintf('leverage of kth shape\n')
    for bi = 1:10
        fprintf('Shape %d: %.2f ± %.2f (p = %f)\n',bi,glm_w(bi,2),glm_w_se(bi,2),glm_p(bi,2))
    end
    hold on
    xlim([-50 550])
    switch id
        case 1
            ylim([-0.5 2])
        case 2
            ylim([-2 10])
    end
    set(gca,'FontSize',24,'FontWeight','bold','Box','off','TickDir','out','XTick',0:200:400)
    set(gca,'XTick',0:100:500,'XTickLabel',[{'0'},{''},{'0.2'},{''},{'0.4'},{''}])
    % *********** Backward Analysis ***********
    
    LLR_back = LLR{2}/10;
    shift_size = 10;
    bin_size = 20;
    num_bin = 100;
    woe_set = -9:2:9;
    clearvars glm_w glm_w_se glm_w_ci glm_w_lo glm_w_hi glm_p
    ri = 1;
    for ti = 1:num_bin
        
        pick_before = logical(lastStim2Sac>=0);
        pick_after = logical(lastStim2Sac-bin_size>=0);
        pickDiff = logical((pick_before-pick_after)==1);
            % remove shapes for the next iteration
        lastStim2Sac = lastStim2Sac - shift_size;
        pick_res = logical((lastStim2Sac - shift_size)>=0);
        pick_remove = logical((pick_before - pick_res)==1);
        lastStim2Sac(pick_remove) = lastStim2Sac(pick_remove) + stim_int;

        pick = logical(pickDiff & isfinite(LLR_back(:,1)));

        LLR_picked = LLR_back(pick,1);
        resLLR_picked = nansum(LLR_back(pick,2:end),2); % sum of logLR for the rest of shapes
        choice_picked = choice(pick);

        [logitCoef,dev,stats] = glmfit([LLR_picked, resLLR_picked],choice_picked-1,'binomial','link','logit','constant',const);
            % Convert the log base from exp to 10
        logitCoef = logitCoef/log(10);
        stats.se = stats.se/log(10);

        glm_w(ti,:) = logitCoef;
        glm_w_se(ti,:) = stats.se;
        glm_w_ci(ti,:) = stats.se * norminv(0.975,0,1);
        glm_w_lo(ti,:) = logitCoef - stats.se * norminv(0.975,0,1);
        glm_w_hi(ti,:) = logitCoef + stats.se * norminv(0.975,0,1);
        glm_p(ti,:) = stats.p;
            
            % shift the matrix for trials that were picked and replace with NaN.
        LLR_back(pick_remove,:) = circshift(LLR_back(pick_remove,:),[0 -1]);
        LLR_back(pick_remove,end) = nan;
        
    end

    figure(fig);hold on
    subplot('position',[0.11+0.8*6/11,0.1,0.8/11*5,0.85]);hold on
    t_bin_center = bin_size/2 + shift_size.*((1:num_bin)-1);
    t_bin_leading_edge = shift_size.*((1:num_bin)-1);
    t_bin_lagging_edge = bin_size + shift_size.*((1:num_bin)-1);
    
        % plot at center of the bin 
    fillTrace(-t_bin_center,squeeze(glm_w(:,offset_flag+1)),squeeze(glm_w_se(:,offset_flag+1)),0.2*[1,1,1]);
    hold on
    
    xlim([-500 0]);
    switch id
        case 1
            ylim([-0.5 2])
        case 2
            ylim([-2 10])
    end
    set(gcf,'position',[500 100 400 300],'color','w');
    set(gca,'FontSize',24,'FontWeight','bold','Box','off','TickDir','out','XTick',-400:200:0,'YTick',[],'YAxisLocation','right')
    set(gca,'XTick',-500:100:0,'XTickLabel',[{''},{'-0.4'},{''},{'-0.2'},{''},{'0'}])
    
    %%%%%% save info %%%%%%
    info.glm_w_end    = glm_w;
    info.glm_w_se_end = glm_w_se;
    info.glm_w_ci_end = glm_w_ci;
    info.glm_w_lo_end = glm_w_lo;
    info.glm_w_hi_end = glm_w_hi;
    info.glm_p_end    = glm_p;

    fig = fig+2;
    
end

%% N* histogram (Fig. 2C)
if fig_switch(3)

    max_num_accum = max(num_accum);
    n_accum = histc(num_accum,1:max_num_accum);
    
    figure(fig)
    AH = bar(1:max_num_accum,n_accum);
    set(AH,'Facecolor','k');
    mean_num_accum = mean(num_accum);
    std_num_accum = std(num_accum);
    fprintf('mean N* = %.1f ± %.1f\n\n',mean_num_accum,std_num_accum)
    xlim([0 17])
    switch id
        case 1
            ylim([0 9000])
            set(gca,'YTick',(0:2:8)*1e3);
        case 2
            ylim([0 7000])
            set(gca,'YTick',(0:2:6)*1e3);
    end
            
    set(gca,'XTick',0:5:15,'XTickLabel',0:5:15);
    set(gca,'FontSize',32,'FontWeight','bold','Box','OFF','TickDir','out');
    
    %%%%%%% save info %%%%%%
    info.n_accum = n_accum;
    
    fig = fig+1;
    
end

%% Psychometric function (Fig. 2D)
if fig_switch(4)

    for ci = 1 % For monkey J, 1: All trials, 2: Red-in-RF only, 3: Green-in-RF trials only 
        switch ci
            case 1 % all in RF
                totalLLR = nansum(LLR{1},2);
                choice_picked = choice;
                plot_color = 'k';
            case 2 % red in RF
                pick = logical(TM(:,9)==1);
                totalLLR = nansum(LLR{1}(pick,:),2);
                choice_picked = choice(pick);
                plot_color = 'r';
            case 3 % green in RF
                pick = logical(TM(:,9)==2);
                totalLLR = nansum(LLR{1}(pick,:),2);
                choice_picked = choice(pick);
                plot_color = 'g';
        end
        
        % To plot the psychometric curve, count the number of choice A (choice 2)
        % for each value of the total logLR
        LLR_edges = floor(min(totalLLR)):ceil(max(totalLLR));
        LLR_all_dist = histc(totalLLR,LLR_edges);
        LLR_choice2 = totalLLR(choice_picked==2,:);
        LLR_choice2_dist = histc(LLR_choice2,LLR_edges);
        p_choice2 = LLR_choice2_dist./LLR_all_dist;
        Y = [LLR_choice2_dist,LLR_all_dist]; % 1st col: # of choice A, 2nd column # of total trials

        [logitCoef,dev,stats] = glmfit(totalLLR,choice_picked-1,'binomial','link','logit');
        logitCoef = logitCoef/log(10);
        stats.se = stats.se/log(10);

        uniqueAbsLLR = unique(abs(LLR_edges));
        for i = 1:length(uniqueAbsLLR)
            bound = uniqueAbsLLR(i);
            unbounded_ind = logical(LLR_edges>=-bound & LLR_edges<=bound);
            unbounded_fraction(i) = sum(Y(unbounded_ind,2))./sum(Y(:,2));
        end
        
        p_pred = 1./(1+10.^(-(logitCoef(2).*LLR_edges + logitCoef(1))))';

        mean_y = Y(:,1)./Y(:,2);
        sem_y  = sqrt(p_pred(i).*(1-p_pred(i))./Y(:,2));
        
            % Compute the expected fraction of correct trials given a single shape (reported in the paper)
        shape_prob = makeStimProb(id,0); % sampling distribution (see Fig. 1B)
        mean_woe = shape_prob*[-9:2:-3,0,0,3:2:9]'/10; % assigned logLR
        p_correct_single = 1/(1+10^(-mean_woe));
        
            % Compute the fraction of correct (rewarded) trials (reported in the paper)
        p_correct = sum(correct)/sum(Y(:,2));
            
            % Compute the fraction of rational trials 
            % (i.e. choosing the target supported by cumulative logLR) (reported in the paper)
        pick_neg = logical(LLR_edges<0);
        pick_zero = logical(LLR_edges==0);
        pick_pos = logical(LLR_edges>0);
        n_rational = sum(Y(pick_neg,2) - Y(pick_neg,1)) + sum(Y(pick_pos,1));
        n_total = sum(Y(:,2))- Y(pick_zero,2);
        p_rational = n_rational/n_total;
        
                
        fprintf('Expected fraction of correct trials given a single shape\n')
        fprintf('p = %.2f\n\n',p_correct_single);
        
        fprintf('Fraction of correct (rewarded) trials\n')
        fprintf('p = %.2f\n\n',p_correct);
        
        fprintf('Fraction of trials in which the monkey chose the target supported by cumulative logLR\n')
        fprintf('p = %.2f\n\n',p_rational);
        
        figure(fig);clf;hold on
        set(gcf,'position',[100 100 700 700],'color','w');
        font_size = 24;
        

        subplot('position',[0.3 0.45 0.6 0.5]); hold on
        LLRaxis = LLR_edges'/10;
        plot(LLRaxis,mean_y,'ko','MarkerFaceColor',plot_color,'MarkerSize',8);
        % plot(LLR_edges/10,p_pred,'-','color',plot_color);
        xlim([-3 3])
        ylim([0 1])

        ylabel('Proportion of choice A','FontSize',font_size,'FontWeight','bold')
        set(gca,'FontSize',font_size,'FontWeight','bold','Box','OFF','TickDir','out');
        hold off
    end
        % trial counts for the psychometric curve
    subplot('position',[0.3 0.2 0.6 0.15])
    bar(LLRaxis,Y(:,2))
    xlim([-3 3])
    xlabel('Evidence for target A (LLR)','FontSize',font_size,'FontWeight','bold')
    ylabel('Trial count','FontSize',font_size,'FontWeight','bold')
    set(gca,'FontSize',font_size,'FontWeight','bold','Box','OFF','TickDir','out');
    
    %%%%%%% save info %%%%%%
    info.uniqueLLR = LLR_edges';
    info.NA = Y;
    info.PA = mean_y;
    
    
    fig = fig+1;
    
end

%% Cumulative evidence at the end of decision (Fig. 2E)
if fig_switch(5)
    num_fig = 1;
    
    shape_prob = makeStimProb(id,0);
    p{1} = 1;
    A = 9999;
    if 1
        for i = 1:20
            WOE = -0.9*i:0.2:0.9*i;
            p{i+1} = conv(p{i},shape_prob);
            pick = find(WOE>=A);
            p_upper_bound(i+1) = sum(p{i+1}(pick));
            pick = find(WOE<=-A);
            p_lower_bound(i+1) = sum(p{i+1}(pick));
            pick = find(WOE>=A|WOE<=-A);
            p{i+1}(pick) = 0;
            pick = find(isfinite(WOE)); % Always choosing the reward-assigned target (unrealistic)
            P = p{i+1}(pick)/sum(p{i+1}(pick));
            varWOE(i) = sum(WOE(pick).^2.*P)-(sum(WOE(pick).*P)).^2;
            stdWOE(i) = sqrt(varWOE(i));
            meanWOE(i) = sum(WOE(pick).*P);
            p_correct_WOE = 10.^WOE./(1+10.^WOE);
            p_correct_WOE = abs(p_correct_WOE-0.5) + 0.5;    
            p_correct(i+1) = sum(p_correct_WOE.*p{i+1});
        end
    end
    
        % preallocation
    cumLLR_all = cell(1,20);
    cumLLR_correct = cell(1,20);
    cumLLR_error = cell(1,20);
    meanCumLLR = nan(20,20,2);
    stdCumLLR = nan(20,20,2);
    seCumLLR = nan(20,20,2);
    
    max_num_accum = max(num_accum);
        % Set separate_plot_flag as follows:
        % 1: Fig. 2E or Fig. S1B -- plot separately for choie A and B, 
        % 0: Fig. S1A -- plot combining both choices (by reversing the sign of weight for choice B)
    separate_plot_flag = 1;
    
    sortBy = 'choice';
    
        % rew indicates reward-assigned target (1:Target B, 2:Target A)
    switch id
        case 1
            rew = rew_targ;
        case 2
            rew = rew_color;
    end
   
    for i = 1 % 1:all trials, 2:correct trials, 3:error trials
        clearvars cumLLR_bigM1 cumLLR_bigM2
        cumLLR_bigM1 = [];
        cumLLR_bigM2 = [];
        for j = 1:2 % index for target (1:Target B, 2:Target A)
            for k = 1:18; % the range of epochs to plot
                if separate_plot_flag
                    switch sortBy
                        case 'choice' % conditioned on choice (Fig. 2E)
                            switch i
                                case 1
                                    pick = find(choice==j & num_accum==k);
                                case 2
                                    pick = find(choice==j & num_accum==k & correct);
                                case 3
                                    pick = find(choice==j & num_accum==k & ~correct);
                            end
                        case 'reward' % conditioned on reward assignment (Fig. S1B)                        
                            switch i
                                case 1
                                    pick = find(rew==j & num_accum==k);
                                case 2
                                    pick = find(rew==j & num_accum==k & correct);
                                case 3
                                    pick = find(rew==j & num_accum==k & ~correct);
                            end   
                    end
                    
                    % Do not plot if the data point consists of less
                        % than 5 trials.
                    if length(pick)<5
                        continue
                    end
                    
                    pickedCumLLR = cumLLR{1}(pick,:)/10;
                    for ei = 1:k
                            % Excluding trials with trump shapes, if any
                        pick_finite = find(pickedCumLLR(:,ei)>-900 & pickedCumLLR(:,ei)<900);
                        if ei==k
                            endCumLLR = pickedCumLLR(pick_finite,ei);
                            
                                % For regression, store in different matrices. 
                            switch j
                                case 1
                                    cumLLR_bigM1 = [cumLLR_bigM1;[ones(size(endCumLLR))*ei,endCumLLR]];
                                case 2
                                    cumLLR_bigM2 = [cumLLR_bigM2;[ones(size(endCumLLR))*ei,endCumLLR]];
                            end
                        end
                        
                            % Do not plot if the data point consists of less
                            % than 5 trials.
                        if length(pick)<5
                            continue
                        end
                        meanCumLLR(k,ei,j) = mean(pickedCumLLR(pick_finite,ei),1);
                        varCumLLR(k,ei,j) = nanvar(pickedCumLLR(pick_finite,ei));
                        stdCumLLR(k,ei,j) = nanstd(pickedCumLLR(pick_finite,ei));
                        seCumLLR(k,ei,j) = nanse(pickedCumLLR(pick_finite,ei));
                    end
                    
                    switch j
                        case 1
                            plot_color = 'b';
                            x_offset = 0.05;
                            plot_marker = 's';
                        case 2
                            plot_color = 'r';
                            x_offset = -0.05;
                            plot_marker = 'o';
                    end
                    
                    if k<=10
                        figure(fig); hold on
                        %%% SEM %%%
                        h1=ploterr(k+x_offset,meanCumLLR(k,k,j),[],stdCumLLR(k,k,j),1,'o','abshhy',0);
                        % h1 = ploterr(0:k,[0;meanCumLLR(1:k,j)],[],[0;seCumLLR(1:k,j)],1,'r-.','abshhy',0.1);
                        set(h1(1),'Marker',plot_marker,'MarkerSize',10,'color','k','MarkerFaceColor','k')
                        set(h1(2),'color','k','LineWidth',0.5)
                    end
                    
                                        
                else % two types of choices combined together
                    
                    cumLLRtemp = cumLLR{1};
                        % if left (green) was chosen, flip the sign of LLR.
                    switch sortBy
                        case 'choice'
                            cumLLRtemp(choice==1,:) = -cumLLRtemp(choice==1,:);
                        case 'reward'
                            cumLLRtemp(rew==1,:) = -cumLLRtemp(rew==1,:);
                    end
                    switch i
                        case 1
                            pick = find(num_accum==k);
                        case 2
                            pick = find(num_accum==k & correct); % choice + shape num + correct
                        case 3
                            pick = find(num_accum==k & ~correct); % choice + shape num + correct
                    end
                    
                        % Do not plot if the data point consists of less
                        % than 5 trials.
                    if length(pick)<5
                        continue
                    end
                    pickedCumLLR = cumLLRtemp(pick,:)/10;
                    j = 1;
                    for ei = 1:k
                            % Excluding trials with trump shapes, if any
                        pick_finite = find(pickedCumLLR(:,ei)>-900 & pickedCumLLR(:,ei)<900);
                        if ei==k
                            endCumLLR = pickedCumLLR(pick_finite,ei);
                            switch i
                                case 1
                                    cumLLR_all{j,k}  = endCumLLR;
                                case 2
                                    cumLLR_correct{j,k}  = endCumLLR;
                                case 3
                                    cumLLR_error{j,k}  = endCumLLR;
                            end
                                % For regression, store in different matrices. (cumLLR_bigM1 = cumLLR_bigM2)
                            cumLLR_bigM1 = [cumLLR_bigM1;[ones(size(endCumLLR))*ei,endCumLLR]];
                            cumLLR_bigM2 = [cumLLR_bigM2;[ones(size(endCumLLR))*ei,endCumLLR]];
                        end
                        meanCumLLR(k,ei,j) = mean(pickedCumLLR(pick_finite,ei),1);
                        varCumLLR(k,ei,j) = nanvar(pickedCumLLR(pick_finite,ei));
                        stdCumLLR(k,ei,j) = nanstd(pickedCumLLR(pick_finite,ei));
                        seCumLLR(k,ei,j) = nanse(pickedCumLLR(pick_finite,ei));
                    end
                    
                    figure(fig); hold on
                    h = ploterr(k,meanCumLLR(k,k,j),[],seCumLLR(k,k,j),1,'o','abshhy',0);
                    set(h(1),'color','k');
                    set(h(2),'color','k');
                    plot((0:k),[0,meanCumLLR(k,1:k,j)],'-o','color','k','MarkerSize',4,'MarkerFaceColor','k');
                    plot(k,meanCumLLR(k,k,j),'o','color','k','MarkerFaceColor','k','MarkerSize',10);
                end
            end            
            
            
                % Model 0: flat bound model
            init_beta = [0.1 0];
            [beta_fit0,err0,exitFlag0,oput0,grad0,hessian0] = fminunc(@(beta) linearFitErr([beta(1) 0],(1:max_num_accum)',diag(meanCumLLR(:,:,j)),diag(seCumLLR(:,:,j))),init_beta);
            
                % Model 1: random stopping model (RT independent of total logLR)
            y = diag(meanCumLLR(:,:,j));
            y_pred = meanWOE'*(-1)^j;
            y_se = diag(seCumLLR(:,:,j));
            pick = logical(abs(y)>0);
            err1 = sum(((y(pick)-y_pred(pick))./y_se(pick)).^2./2);
            
            ERR0(i,j) = err0;
            ERR1(i,j) = err1;
            n_data(i,j) = sum(pick);
            endCumLLR_beta(i,j,:) = beta_fit0;
                
        end
        
            % AIC and BIC analysis for Fig. S1B
            % AIC0 - AIC1: if this is negative, model 0 (flat bound) is more likely. 
        delta_AIC(i) = 2*(1-0) - 2*(sum(-ERR0(i,:))-sum(-ERR1(i,:)));
        num_data = sum(n_data(i,:));
        delta_BIC(i) = (1*(log(num_data)+log(2*pi))-0) - 2*(sum(-ERR0(i,:))-sum(-ERR1(i,:)));
        
        fprintf('delta AIC = %d\n\n',round(delta_AIC(i)));
        fprintf('delta BIC = %d\n\n',round(delta_BIC(i)));
        
        figure(fig)
        set(gcf,'position',[100 200 800 500],'color','w');
        xlim([0 11])
        if separate_plot_flag
            ylim([-3 3])
        else
            ylim([0 3])
        end
        set(gca,'FontSize',28,'FontWeight','bold','Box','OFF','TickDir','out')
        
            % Is the regression slope for cumulative logLR at the end of 
            % decision significantly different from zero? (weighted regression)
        for j = 1:2
            tempM = eval(['cumLLR_bigM',num2str(j)]);
            unique_N = round(unique(tempM(:,1)));
            bigS = nan(size(tempM,1),1);
            for ni = 1:max(unique_N)
                pick_n = logical(round(tempM(:,1))==ni);
                std_n(j,ni) = std(tempM(pick_n,2));
                bigS(pick_n) = std_n(j,ni);
            end
            [betaGLM(:,j),devGLM,stats]=glmfit(tempM(:,1),tempM(:,2),'normal','weights',bigS.^-2);
            betaGLM_se(:,j) = stats.se;
            betaGLM_p(:,j) = stats.p;
            
            fprintf('slope for cumulative logLR at N*: %.2f ± %.2f (logLR/epoch) (p = %f)\n\n',betaGLM(2,j),betaGLM_se(2,j),betaGLM_p(2,j));
            
            N_axis = 1:10;
            pred_cumWoe = betaGLM(2,j)*N_axis + betaGLM(1,j);
            if separate_plot_flag
                switch sortBy
                    case 'choice'
                        plot(N_axis,pred_cumWoe,'--','color','k')
                    case 'reward'
                        plot(N_axis,endCumLLR_beta(i,j,1)*ones(1,length(N_axis)),'-','color','k')
                        plot(N_axis,meanWOE(N_axis)*(-1)^j,'--','color','k')
                end
            end
            %xlabel('Number of shapes used for decision','FontSize',28,'FontWeight','bold'); 
            %ylabel('Total evidence (logLR)','FontSize',28,'FontWeight','bold');
            set(gca,'FontSize',28,'FontWeight','bold','Box','OFF','TickDir','out')
        end
        
        %%%%%% save info %%%%%%
        switch i
            case 1
                info.cumLLR_all = cumLLR_all;
                info.meanCumLLR = meanCumLLR;
                info.stdCumLLR  = stdCumLLR;
                info.seCumLLR   = seCumLLR;
            case 2
                info.cumLLR_correct = cumLLR_correct;
                info.meanCumLLR_correct = meanCumLLR;
                info.stdCumLLR_correct  = stdCumLLR;
                info.seCumLLR_correct   = seCumLLR;
            case 3
                info.cumLLR_error = cumLLR_error;
                info.meanCumLLR_error = meanCumLLR;
                info.stdCumLLR_error  = stdCumLLR;
                info.seCumLLR_error   = seCumLLR;
        end
        
    end
    fig = fig+1;
end

%% Subjective weight for each shape across epochs
if fig_switch(6)
    
    EM = zeros(size(TM,1),total_shape);
    for ind = 1:size(TM,1)
        shapes = shapeMtrx(ind,1:num_accum(ind));
        EM(ind,:) = histc(shapes,(1+shape_offset):10);
    end
    
    fprintf('compute SWOE\n\n');
        % compute the subjective weight of evidence (Fig. 2F, see also Equation 3)
    [shapeSWOE,dev,stats] = glmfit(EM,choice-1,'binomial','link','logit','constant','off');
        % convert the log base from exp to 10
    shapeSWOE = shapeSWOE/log(10);
    shapeSWOE_se = stats.se/log(10);
    
    x_fit = shapeSWOE;
    stdErr = shapeSWOE_se;
    for i = 1:total_shape
        if i<= total_shape/2
            shapeSWOE(i) = x_fit(2*i);
            shapeSWOE_se(i) = stdErr(2*i);
        else
            shapeSWOE(i) = x_fit(total_shape-2*(i-total_shape/2)+1);
            shapeSWOE_se(i) = stdErr(total_shape-2*(i-total_shape/2)+1);
        end
    end
    
    figure(fig)
    clf;hold on
    
    switch id
        case 1
            woe_axis = -[-9:2:-3,3:2:9]*0.1;
        case 2
            woe_axis = [-9:2:-3,3:2:9]*0.1;
    end
    
    [rho,pval] = corr(woe_axis',shapeSWOE,'type','Spearman');
    fprintf('r = %.2f (p = %f)\n',rho,pval)

    h = ploterr(woe_axis,shapeSWOE,[],shapeSWOE_se,0.5,'ko','abshhy',0);
    set(h(1),'MarkerSize',8,'MarkerFaceColor','k');

    xlim([-1.1 1.1])
    switch id
        case 1
            ylim([-1 1.2])
        case 2
            ylim([-6 4])
    end

    font_size = 24;
    xlabel('True Weight (LLR)','FontSize',font_size,'FontWeight','bold'); 
    ylabel('Subjective Weight (LLR)','FontSize',font_size,'FontWeight','bold');
    set(gca,'FontSize',font_size,'FontWeight','bold','Box','OFF','TickDir','out')
    set(gcf,'position',[200 200 500 450])
    hold off
    
    info.shapeSWOE = shapeSWOE;
    info.shapeSWOE_se = shapeSWOE_se;
    
    fig = fig + 1;
end