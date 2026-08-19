function varargout = fillRect(A,B,varargin)
    % Filling a rectangle with a specified color
    % A = [Ax,Ay] is a lower left corner 
    % B = [Bx,By] is a upper right corner
    % c is a filling color
    if isempty(varargin)
        c = [0.8 1 1];
    else
        c = varargin{1};
    end
    M = [A(1),A(2);
         B(1),A(2);
         B(1),B(2);
         A(1),B(2);
         A(1),A(2)];
    varargout{1} = fill(M(:,1),M(:,2),c);
    return
end