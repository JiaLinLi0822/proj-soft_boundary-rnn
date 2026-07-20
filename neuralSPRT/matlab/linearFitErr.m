function err = linearFitErr(beta,x,y,y_se)
    % if beta is a scalar, fit a slope without an offset.
    % if beta is a vector with two elements, fit a slope with an offset.
    
    pick = logical(y_se>0);
    x = x(pick);
    y = y(pick);
    y_se = y_se(pick);
    
    if length(beta)<2
        y_pred = x*beta;
        err = sum(((y-y_pred)./y_se).^2./2);
    else
        y_pred = beta(1) + x*beta(2);
        err = sum(((y-y_pred)./y_se).^2./2);
    end
    % n = length(y);
    % err = sum(((y-y_pred)./y_se).^2./2) + sum(log(y_se)) + n/2*log(2*pi);
end
