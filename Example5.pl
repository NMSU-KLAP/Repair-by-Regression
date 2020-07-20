% Example 5

fluents([inR1,inR2,lightR1,lightR2]).
dynamicLaw(leave,[inR1],[inR2]).
dynamicLaw(leave,[inR2],[inR1]).
dynamicLaw(turnOn,[lightR1],[inR1]).
dynamicLaw(turnOn,[lightR2],[inR2]).
senseAct(senseLightR1,lightR1).
senseAct(senseLightR2,lightR2).
exeCnd(leave,[]).
exeCnd(senseLightR1,[inR1]).
exeCnd(senseLightR2,[inR2]).
exeCnd(turnOn,[]).
staticLaw([inR2],[-inR1]).
staticLaw([inR1],[-inR2]).
staticLaw([-inR2],[inR1]).
staticLaw([-inR1],[inR2]). 
