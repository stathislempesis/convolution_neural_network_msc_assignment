function [state, reward] = pickup( state )
%PICKUP Summary of this function goes here
%   Detailed explanation goes here

taxiloc=state(1);
passloc=state(2);

reward=-1;
if passloc<5 && getdroplocation(passloc)==taxiloc,
    state(2)=5;
    reward=1;
end

end

