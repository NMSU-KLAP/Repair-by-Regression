fluents([in001,in002,in003,in004,in005,in006,in007,on001,on002,on003,on004,on005,on006,on007]).
dynamicLaw(move,[in002],[in001]).
dynamicLaw(move,[in003],[in002]).
dynamicLaw(move,[in004],[in003]).
dynamicLaw(move,[in005],[in004]).
dynamicLaw(move,[in006],[in005]).
dynamicLaw(move,[in007],[in006]).
senseAct(sense001,on001).
senseAct(sense002,on002).
senseAct(sense003,on003).
senseAct(sense004,on004).
senseAct(sense005,on005).
senseAct(sense006,on006).
senseAct(sense007,on007).
exeCnd(move,[]).
exeCnd(sense001,[in001]).
exeCnd(sense002,[in002]).
exeCnd(sense003,[in003]).
exeCnd(sense004,[in004]).
exeCnd(sense005,[in005]).
exeCnd(sense006,[in006]).
exeCnd(sense007,[in007]).
staticLaw([-in001],[in002]).
staticLaw([-in001],[in003]).
staticLaw([-in001],[in004]).
staticLaw([-in001],[in005]).
staticLaw([-in001],[in006]).
staticLaw([-in001],[in007]).
staticLaw([-in002],[in001]).
staticLaw([-in002],[in003]).
staticLaw([-in002],[in004]).
staticLaw([-in002],[in005]).
staticLaw([-in002],[in006]).
staticLaw([-in002],[in007]).
staticLaw([-in003],[in001]).
staticLaw([-in003],[in002]).
staticLaw([-in003],[in004]).
staticLaw([-in003],[in005]).
staticLaw([-in003],[in006]).
staticLaw([-in003],[in007]).
staticLaw([-in004],[in001]).
staticLaw([-in004],[in002]).
staticLaw([-in004],[in003]).
staticLaw([-in004],[in005]).
staticLaw([-in004],[in006]).
staticLaw([-in004],[in007]).
staticLaw([-in005],[in001]).
staticLaw([-in005],[in002]).
staticLaw([-in005],[in003]).
staticLaw([-in005],[in004]).
staticLaw([-in005],[in006]).
staticLaw([-in005],[in007]).
staticLaw([-in006],[in001]).
staticLaw([-in006],[in002]).
staticLaw([-in006],[in003]).
staticLaw([-in006],[in004]).
staticLaw([-in006],[in005]).
staticLaw([-in006],[in007]).
staticLaw([-in007],[in001]).
staticLaw([-in007],[in002]).
staticLaw([-in007],[in003]).
staticLaw([-in007],[in004]).
staticLaw([-in007],[in005]).
staticLaw([-in007],[in006]).
history([(sense001,[[on001]]),(move,[]),(sense002,[[on002]]),(move,[]),(sense003,[[on003]]),(move,[]),(sense004,[[on004]]),(move,[]),(sense005,[[on005]]),(move,[]),(sense006,[[on006]]),(move,[]),(sense007,[[-on007]])]).
initially([[in001,on001,on002,on003,on004,on005,on006,on007]]).
