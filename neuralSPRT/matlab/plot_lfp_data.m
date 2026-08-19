function plot_lfp_data(info)

    if ~exist('info','var')
        load('dataE_LFP.mat','info')
    end

    trial_id = 1;

    trialLFP = info.LM{trial_id};
    trialTime = info.LMe{trial_id}(:,1);
    trialEvents = info.LMe{trial_id}(:,2);
    trial_v = info.TM(trial_id,:);

    ecodeRTshape;

    %% Plot trial events and spike rasters
    figure(1);clf;
    subplot(3,1,1);hold on;
    for i = 1:length(trialEvents)
        if ~any(trialEvents(i)==[E_SPIKE,E_FP_OFF,E_FP2_ON,E_STIM_OFF,E_CLEAR_SCREEN,...
                E_SHAPE_TASK,E_PARAM,E_FEEDBACK_ON,E_TRIAL_INFO,E_TARGET1_ACQUIRED,E_TARGET2_ACQUIRED])
            en = event_name(event_code==trialEvents(i),:);
            en = strrep(en,'E_','');
            en = strrep(en,'E-','');
            en = strrep(en,'STIM','SHAPE');
            text(trialTime(i),2,en,'Rotation', 90);            
        end
    end
    spikeTime = trialTime(trialEvents==E_SPIKE);
    tickRaster(spikeTime,1,'k')
    xlim([trialTime(1),trialTime(end)])
    ylim([0,3])
    xlabel('Elapsed trial time (ms)')
    set(gca,'YTick',1:2,'YTickLabel',{'Spikes','Event Name'})

    %% Plot cumulative evidence in a trial
    t_stim_on = trialTime(trialEvents==E_STIM_ON);
    n_shape_on = length(t_stim_on);
    t_stim_on = [0;t_stim_on;trialTime(trialEvents==E_SACCADE)];
    woe = [inf, -inf, 0.9, -0.9, 0.7, -0.7, 0.5, -0.5, 0.3, -0.3, 0.1, -0.1]; % weigt of evidence (logLR)

    n_shape_used = sum(isfinite(trial_v(11:30)));
    w = [0,woe(int8(trial_v(11:11+n_shape_used-1)))];
    cum_w = cumsum(w);

    subplot(3,1,2);hold on;
    for si = 1:n_shape_used+1
        plot(t_stim_on(si:si+1),cum_w(si)*[1,1],'k-');
    end
    xlim([trialTime(1),trialTime(end)])
    ylim([-2 2])
    xlabel('Elapsed trial time (ms)')
    ylabel('Cumulative evidence (logLR)')

    %% Plot LFP
    subplot(3,1,3);hold on;
    plot(trialLFP);
    xlim([trialTime(1),trialTime(end)])
    xlabel('Elapsed trial time (ms)')
    ylabel('LFP (a.u.)')

end

function tickRaster(spikeTime,trialNum,color)
    if size(spikeTime,1)~=1
        spikeTime = spikeTime';
    end
    
    s = [repmat(spikeTime,2,1);nan(1,length(spikeTime))];
    t = [zeros(1,length(spikeTime))+trialNum-0.3;...
         zeros(1,length(spikeTime))+trialNum+0.3;...
         nan(1,length(spikeTime))];
    plot(s,t,'-','color',color);
end