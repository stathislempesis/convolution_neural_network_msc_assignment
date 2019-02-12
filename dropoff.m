function [ state, reward ] = dropoff( state, iteration )
%DROPOFF Summary of this function goes here
%   Detailed explanation goes here

taxiloc=state(1);
passloc=state(2);
droploc=state(3);

reward=-1;
if passloc==5 && taxiloc==getdroplocation(droploc),
    state(2)=droploc;
    reward=10/iteration;
end

end
