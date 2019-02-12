% Q-learning: the taxi problem

% number of states
ww = 5;  % width of grid world
wh = 5;  % length of grid world

% number of drop off locations
Sdrop=4;

% number of taxi locations
Staxi=ww*wh;

% number of passanger locations
Spass=5;

% number of states of the problem
S = [Staxi, Spass, Sdrop];

% number of actions
A = 6;    % actions: U, D, L, R, Pick and Drop
% total number of learning trials
T = 20000;

%initialisation
Q = 2*rand([S A]);
V = max(Q,[],4);
eta = 0.2;
gamma = 0.9;
epsilon = 1.0;

% run the algorithm for T trials
for i=1:T,
    
    	% set goal
    	Gdrop=randi(Sdrop);
    	Goal=[getdroplocation(Gdrop),Gdrop,Gdrop];
    
    	% set the starting state
    	s0=[randi(Staxi),randi(Spass),Gdrop];
    
    	pickup=-1;

    	average_r = 0;

    	% each trial consists of re-inialisation and a S*S moves
    	% a random walker will reach the goal in a number of steps proportional to
    	% S*S
    	for u=1:1000,

        	[V(s0(1),s0(2),s0(3)),a0]= max(Q(s0(1),s0(2),s0(3),:));  % we only need the a0 here

        	% exploration (epsilon-greedy)
        	if (rand(1)<epsilon), a0=randi(A); end;

       	 	% now moving left, right, up, down, or not
        	% and don't step outside the track
        	if (pickup < 0),
            		[s1, r]=applyaction(s0, a0);
            	if (a0==5 && r >= 0), pickup=u; end
		else
            		[s1, r]=applyaction(s0, a0, u-pickup);
        	end

		average_r = average_r + r;
        
        	% now the learning step
        	V(s1(1),s1(2),s1(3))=max(Q(s1(1),s1(2),s1(3),:));
        	Q(s0(1),s0(2),s0(3),a0)= (1-eta)*Q(s0(1),s0(2),s0(3),a0) + eta*(r+gamma*V(s1(1),s1(2),s1(3)));

        	% goto next trial once the goal is reached
        	if (all(s0==Goal)), break; end
        	s0=s1;
    end

    % average reward
    fprintf('The average reward is %f\n', average_r/u);

    % exploration rate gets lower and lower
    % (note that this was wrong in the 1D version: a "T" instead of the "t")
    epsilon=1/sqrt(i);
    % we could also reduce the learning rate
    % eta=1/sqrt(t);

    % plotting the Q-function
    if (rem(i,10)==0)
        Qflt=reshape(V,25*5,4);
        plot(Qflt);
        hold on
        title(i);
        ylim([0 1/(1-gamma)]);
        plot(Qflt,'ok');
        hold off
        drawnow
    end;
end

state_values = [];

% set goal
Gdrop=randi(Sdrop);
Goal=[getdroplocation(Gdrop),Gdrop,Gdrop];
    
% set the starting state
s0=[randi(Staxi),randi(Spass),Gdrop];

x0=mod(s0(1)-1, 5)+1;
y0=idivide(int32(s0(1)-1), 5)+1;

fprintf('\nThe initial state of taxi is %d,%d\n', x0, y0);

if(s0(2)~=5)

	x0=mod(s0(2)-1, 5)+1;
	y0=idivide(int32(s0(2)-1), 5)+1;

	fprintf('\nThe initial state of passenger is %d,%d\n', x0, y0);

else

	fprintf('\nThe initial state of passenger is in the taxi\n');

end

x0=mod(getdroplocation(Gdrop)-1, 5)+1;
y0=idivide(int32(getdroplocation(Gdrop)-1), 5)+1;

fprintf('\nThe goal state is %d,%d\n', x0, y0);

pickup=-1;

for u=1:1000,

	[V(s0(1),s0(2),s0(3)),a0]= max(Q(s0(1),s0(2),s0(3),:));  % we only need the a0 here

	switch a0
	    case 1, fprintf('\nBest action is north\n');
	    case 2, fprintf('\nBest action is south\n');
	    case 3, fprintf('\nBest action is west\n');
	    case 4, fprintf('\nBest action is east\n');
	    case 5, fprintf('\nBest action is pick up\n');
            case 6, fprintf('\nBest action is drop off\n');
	end

        % now moving left, right, up, down, or not
        % and don't step outside the track
        if (pickup < 0),
            [s1, r]=applyaction(s0, a0);
            if (a0==5 && r >= 0), pickup=u; end
        else
            [s1, r]=applyaction(s0, a0, u-pickup);
        end

        % goto next trial once the goal is reached
        if (all(s1==Goal)), break; end
        s0=s1;

	x0=mod(s0(1)-1, 5)+1;
	y0=idivide(int32(s0(1)-1), 5)+1;

	fprintf('\nNext state of taxi is %d,%d with value %f\n', x0, y0, V(s0(1),s0(2),s0(3)));

	state_values = [state_values V(s0(1),s0(2),s0(3))];
end

bar(state_values)
ylabel('value of state')
xlabel('movement of taxi')
