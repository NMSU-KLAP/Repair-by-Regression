% Example sequence of actions
% Regession: [[-f,-g,-h]]

fluents([f,g,h]).
dynamicLaw(a1,[-f],[-g]).
dynamicLaw(a1,[-g],[-f]).
dynamicLaw(a2,[f],[-f,-g]).
dynamicLaw(a2,[-h],[g]).
senseAct(s,h).
exeCnd(a1,[]).
exeCnd(a2,[]).
exeCnd(s,[]).
staticLaw([h],[f]).
staticLaw([-g],[-h]). 
history([(a1,[]),(s,[[-h]]),(a2,[]),(s,[[h]])]).
