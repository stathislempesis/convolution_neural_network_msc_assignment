function [ state, reward ] = movetaxi( state, move )
%MOVETAXI Summary of this function goes here
%   Detailed explanation goes here

taxiloc=state(1);
x0=mod(taxiloc-1, 5)+1;
y0=idivide(int32(taxiloc-1), 5)+1;

switch move
    case 'N', x1=x0; y1=y0-1;
    case 'S', x1=x0; y1=y0+1;
    case 'W', x1=x0-1; y1=y0;
    case 'E', x1=x0+1; y1=y0;
end

wall0= x1<1 | x1>5 | y1<1 | y1>5;
wall1=(y1==1|y1==2)&(x0==2&x1==3|x0==3&x1==2);
wall2=(y1==4|y1==5)&(x0==1&x1==2|x0==2&x1==1);
wall3=(y1==4|y1==5)&(x0==3&x1==4|x0==4&x1==3);

moved=~(wall0|wall1|wall2|wall3);

if moved,
    state(1)=(y1-1)*5 + x1;
    reward=0;
else
    reward=-1;
end

end
