function [state, reward] = applyaction( state, action, iteration )
%APPLYACTION Summary of this function goes here
%   Detailed explanation goes here

if nargin < 3, iteration=1; end

switch action,
    case 1, % go north
        [state, reward] = movetaxi(state,'N');
    case 2, % go south
        [state, reward] = movetaxi(state,'S');
    case 3, % go west
        [state, reward] = movetaxi(state,'W');
    case 4, % go east
        [state, reward] = movetaxi(state,'E');
    case 5, % pick up passanger
        [state, reward] = pickup(state);
    case 6, % drop off passanger
        [state, reward] = dropoff(state, iteration);
end

end

