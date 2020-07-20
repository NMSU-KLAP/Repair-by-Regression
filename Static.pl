% Example with s.c.l.
% Regression: [[f,g,h],[h,-f,-g]]

fluents([f,g,h]).
dynamicLaw(a,[f],[-f]).
dynamicLaw(a,[-f],[f]).
dynamicLaw(a,[-h],[f,-g]).
senseAct(s,h).
exeCnd(a,[]).
exeCnd(s,[]).
staticLaw([-g],[-f]).
hystory([(a,[]),(s,[[h]])]).
