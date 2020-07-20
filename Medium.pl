% Example medium size
% Regression: [[f,g,h,n,-m],[f,g,h,-m,-n],[f,g,n,-h,-m],[f,g,-h,-m,-n],[f,m,n,-g],[f,m,-g,-h,-n],[g,n,-f,-h,-m]]
% Satoh: [[f,g,h,n,-m],[f,g,n,-h,-m],[f,m,n,-g],[g,n,-f,-h,-m]]
% Dalal: [[f,g,h,n,-m],[f,g,n,-h,-m],[f,m,n,-g],[g,n,-f,-h,-m]]

fluents([f,g,h,m,n]).
dynamicLaw(a1,[h,-f],[g,m]).
dynamicLaw(a1,[-f],[-h,-n]).
dynamicLaw(a1,[f,h],[-f,-g]).
dynamicLaw(a1,[-h],[f,g,-m]).
dynamicLaw(a2,[n],[-f,-h]).
dynamicLaw(a2,[m,-g],[h]).
dynamicLaw(a2,[g,-f],[-h]).
dynamicLaw(a3,[h,m],[g,n,-f]).
dynamicLaw(a3,[m],[f,g]).
dynamicLaw(a3,[n,-m],[h,-f,-n]).
senseAct(s1,n).
senseAct(s2,g).
exeCnd(a1,[]).
exeCnd(a2,[]).
exeCnd(a3,[]).
exeCnd(s1,[]).
exeCnd(s2,[]).
staticLaw([g],[f,h]).
staticLaw([m],[-g]).
staticLaw([g,n],[-f,-h,-m]). 
initially([[f,g,h,m,n],[f,g,m,n,-h],[g,h,m,n,-f],[g,h,n,-f,-m]]).
history([(a1,[]),(a2,[]),(s1,[[n]]),(a3,[]),(s2,[[g]])]).
