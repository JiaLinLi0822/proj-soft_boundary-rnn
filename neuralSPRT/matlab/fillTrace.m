function varargout = fillTrace(X,Y,err,varargin)
    if isempty(varargin)
        c = 'b';
        fill_c = 'b';
    end
    if length(varargin)>=1
        c = varargin{1};
        fill_c = 1-((1-c).*0.8);
    end
    if length(varargin)>=2
        c_alpha = varargin{2};
        fill_c = 1-((1-c).*c_alpha);
    end
    if isrow(X)
        X = X';
    end
    if isrow(Y)
        Y = Y';
    end
    if isrow(err)
        err = err';
    end
    pick = isfinite(Y);
    
    X = X(pick);
    Y = Y(pick);
    err(isnan(err)) = 0;
    err = err(pick);
    
    Ylow = Y-err;
    Yhi  = Y+err;
    M = [X Ylow;flipud(X),flipud(Yhi)];
    
    % plot them
    hold on
    h = fill(M(:,1),M(:,2),fill_c,'LineStyle','none','FaceAlpha',1);
    % h = patch(M(:,1),M(:,2),fill_c);
    % set(h,'LineStyle','none','FaceAlpha',0.7);
    plot(X,Y,'color',c,'LineWidth',1);
    hold off
    
    varargout{1} = h;
    return
end