% Example without s.c.l.
% Regression: [[f,h,-g],[h,-f,-g]]

fluents([f,g,h]).
dynamicLaw(a,[-f],[g]).
dynamicLaw(a,[-f],[f]).
dynamicLaw(a,[g],[f,-g]).
dynamicLaw(a,[-h],[-g]).
senseAct(s,h).
exeCnd(a,[h]).
exeCnd(s,[]).
staticLaw([],[]). 
history([(a,[]),(s,[[-h]])]).
