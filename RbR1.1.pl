:- use_module(library(apply),[partition/4,maplist/3]).
:- use_module(library(lists)).
:- use_module(library(ordsets)).

% ================================= USAGE =================================
% 0) Always use sorted sets, see sort/2.
%
% 1) Define the domain, for example 
%       dynamicLaw(a,[f],[-g]).
%       staticLaw([h],[-f]).
%       exeCnd(a,[h]).
%
% 2) Define a history, for example
%       H = [(a,[[g,-f],[h]]),(b,[])]
%
% 4) Calculate the regression, for example
%       regress(H,R)

% ================================= INSTANCES =================================

%:- include('Example5.pl').
:- include('Medium.pl').
%:- include('NoStatic.pl').
%:- include('Seq.pl').
%:- include('Static.pl').
%:- include('Test004.pl').
%:- include('Test005.pl').
%:- include('Test006.pl').
%:- include('Test007.pl').
%:- include('Test008.pl').
%:- include('Test009.pl').
%:- include('Test010.pl').

% ================================= PRINT =================================

prettyPrintProgression(Progression) :-    
    format('Progression:~n'),
    prettyPrintDnf(Progression).

prettyPrintRegression(Regression) :-    
    format('Regression:~n'),
    prettyPrintDnf(Regression).
    
prettyPrintRevision(Revision) :-    
    format('Revision:~n'),
    prettyPrintDnf(Revision).
   
prettyPrintDnf(Formula) :-
    findall(SortedCnj,(member(Cnj,Formula),predsort(cmpByFluent,Cnj,SortedCnj)),Formula0),
    sort(Formula0,Formula1),
    prettyPrintDnfBase(Formula1).
    
prettyPrintDnfBase([]) :- !.
prettyPrintDnfBase(Formula) :-    
    Formula = [Cnj|OthersCnj],
    format('  ~w~n',[Cnj]),
    prettyPrintDnfBase(OthersCnj).
    
rawPrintDnf(Formula) :- format('~w~n',[Formula]).
    
% ================================= GENERIC AUXILIARY FUNCTIONS =================================

assertChk(_) :- true.
/*
assertChk(Goal) :-
(
    \+call(Goal) ->
    (
        current_predicate(ansi_format/3) ->
            ansi_format([fg(red)],'ERROR: Assertion "~w" failed!~n',[Goal])
        ;
            format('ERROR: Assertion "~w" failed!~n',[Goal]),
        false 
    );(
        true
    )
).
*/

% Reference: https://github.com/SWI-Prolog/swipl/blob/master/library/apply.pl
foldl(Goal,List,Accumulator,Result) :- foldlBase(List,Goal,Accumulator,Result).

foldlBase([],_,Accumulator,Accumulator).
foldlBase([H|T],Goal,Accumulator,Result) :-
    call(Goal,H,Accumulator,Result0),
    foldlBase(T,Goal,Result0,Result).
    
foldr(Goal,List,Accumulator,Result) :- 
    reverse(List,ReversedList), 
    foldl(Goal,ReversedList,Accumulator,Result).

first((F,_),F).
firsts(ListOfCouples,List) :- maplist(first,ListOfCouples,List).    
    
second((_,S),S).
seconds(ListOfCouples,List) :- maplist(second,ListOfCouples,List).

listPartition([H|T],[H|T1],List2) :- listPartition(T,T1,List2).
listPartition([H|T],List1,[H|T2]) :- listPartition(T,List1,T2).
listPartition([],[],[]).

minimals([],_,[]) :- !.
minimals(List,SmallerPre,Minimals) :- 
    findall(Elem,(member(Elem,List),minimalChk(Elem,SmallerPre,List)),Minimals).  
    
minimalChk(Elem,SmallerPred,List) :- forall((member(Elem_i,List), Elem_i \== Elem),\+call(SmallerPred,Elem_i,Elem)).

smallerBySubset(Set1,Set2) :- 
    ord_subset(Set1,Set2).

sortedChk(List) :- sort(List,List).

% ================================= DOMAIN AUXILIARY FUNCTIONS =================================

% Calculate ClosedSet = Cl(Set).
%
% S: Set of literals (e.g. [f,g,-h])
% C: Set of literals (e.g. [f,g,-h])
closure(Set,ClosedSet) :-  
    contradictionChk(Set), 
    staticLawsChk(Set),
    findall(Eff,(staticLaw(Eff,Pre),ord_subset(Pre,Set),\+ord_subset(Eff,Set)),T),
    ord_union(T,ToAdd),
    (
        ToAdd \== [] -> 
        (
            ord_union(Set,ToAdd,Set0),
            closure(Set0,ClosedSet)
        );(
            ClosedSet = Set
        ) 
    ).

% Compare two literals L1,L2 w.r.t. the relative fluents
%
% R: Comparison result (e.g. <)
% L1: Literal (e.g. -f)
% L2: Literal (e.g. -f)
cmpByFluent(Order,Literal1,Literal2) :-
    fluent(Literal1,Fluent1), 
    fluent(Literal2,Fluent2), 
    compare(Order,Fluent1,Fluent2).    

% Check if S contains two complementary literals
%
% S: Set of literals (e.g. [f,g,-h])
contradictionChk(Set) :- 
    assertChk(sortedChk(Set)), 
    negations(Set,NegatedSet),
    ord_disjoint(Set,NegatedSet).

% Check if S triggers a s.c.l. that makes S contradictory.
%
% S: Set of literals (e.g. [f,g,-h]) 
consistentChk(Set) :-
    closure(Set,ClosedSet),
    once(stateExpansion(ClosedSet,_)).

% Calculate E = e(A,Phi).
%
% Phi: Set of literals (e.g. [f,g,-h])
% A: Ontic action (e.g. a)
% Eff: Set of literals (e.g. [f,g,-h])
e(Act,Phi,Eff) :- 
    findall(E,(dynamicLaw(Act,E,P),ord_subset(P,Phi)),T),
    ord_union(T,Eff). 

% Collect in E the effects of the rules in R.
%
% R: Set of couples effect-precondition (e.g [([f,-g],[-h]),([g,i],[f])])
% E: Set of literals (e.g. [f,g,-h])
e(Rules,Eff) :- 
    firsts(Rules,T), 
    ord_union(T,Eff).

% Check if A is executable from Phi.
%
% A: Ontic action (e.g. a)
% Phi: Set of set of literals (e.g. [[f,g],[i,-h]])   
executableChk(Act,Phi):- 
    exeCnd(Act,Cnd),
    !,
    ord_subset(Cnd,Phi).    

% Calculate Psi1 = Psi ∧ O.
%
% Psi: Set of set of literals (e.g. [[f,g],[i,-h]])
% O: Set of set of literals (e.g. [[f,g],[i,-h]])
% Psi1: Set of set of literals (e.g. [[f,g],[i,-h]])    
forceObservation([],Obs,Obs) :- !.
forceObservation(Psi,[],Psi) :- !.
forceObservation(Psi,Obs,Phi) :- 
    findall(Phi_i,(member(Psi_i,Psi),member(Obs_i,Obs),unionIfConsistent(Psi_i,Obs_i,Phi_i)),Phi0), 
    Phi0 \== [],
    sort(Phi0,Phi).

% Calculate S, minimal superset of Init, such that there in no rules in R which precondition are satisfied.
%
% Init: Set of literals (e.g. [f,g,-h])
% Rules: Set of couples effect-precondition (e.g [([f,-g],[-h]),([g,i],[f])])
% S: Set of literals (e.g. [f,g,-h])    
blockingExpansion(Init,[],Init) :- !.
blockingExpansion(Init,Rules,Set) :-
    Rules = [(_,Pre)|OthersR],
    blockingExpansion(Init,OthersR,Set0),
    negations(Pre,NegatedPre), 
    ord_intersection(Set0,NegatedPre,Intersection,Difference),
    (
        Intersection == [] ->
        (
            member(L,Difference), 
            ord_add_element(Set0,L,Set)
        );(
            Set = Set0
        )
    ).

% Calculate U = S1 ∪ S2 only if it is consistent.
%
% S1: Set of literals (e.g. [f,g,-h])
% S2: Set of literals (e.g. [f,g,-h])
unionIfConsistent(Set1,Set2,Union):-
    ord_union(Set1,Set2,Union), 
    consistentChk(Union).

% Calculate the fluent(s) Fluent relative to the literal(s) Literal.
%
% Literal: Literal (e.g. -f)
% Fluent: Fluent (e.g. f)    
fluent(Literal,Fluent) :- 
    Literal = -X -> 
        Fluent = X 
    ; 
        Fluent = Literal.


fluents(Literals,Fluents) :- 
    maplist(fluent,Literals,Fluents0), 
    sort(Fluents0,Fluents).


% Generate a literal(s) Literal from the fluent(s) Fluent.
%
% Fluent: Fluent (e.g. f)
% Literal: Literal (e.g. -f)
literal(Fluent,Literal) :- 
    Literal = Fluent 
    ; 
    negation(Fluent,Literal).
    
literals(Fluents,Literals) :-  
    maplist(literal,Fluents,Literals0), 
    sort(Literals0,Literals).

% Calculate the negation(s) NegatedLiteral of the literal(s) Literal.
%
% Literal: Literal (e.g. -f)
% NegatedLiteral: Literal (e.g. -f)
negation(Literal,NegatedLiteral) :- 
    Literal = -X -> 
        NegatedLiteral = X 
    ; 
        NegatedLiteral = -Literal.

negations(Literals,NegatedLiterals) :- 
    maplist(negation,Literals,NegatedLiterals0),
    sort(NegatedLiterals0,NegatedLiterals).

% Calculate Complement = F \ Fluents 
%
% Fluents: Set of fluents (e.g. [f,g,h])
% Complement: Set of fluents (e.g. [f,g,h])
complementaryFluents(Fluents,Complement) :-
    fluents(AllFluents), 
    !,
    ord_subtract(AllFluents,Fluents,Complement).


% Calculate Union = Set ∪ -Set
%
% Set: Set of literals (e.g. [f,g,-h])
% Union: Set of literals (e.g. [f,g,-h])
unionWithNegation(Set,Union):-
    negations(Set,NegatedSet), 
    ord_union(Set,NegatedSet,Union).

% Calculate a split (Ra+,Ra-) of Ra compatible with Psi
%
% Ra: Set of couples effect-precondition (e.g [([f,-g],[-h]),([g,i],[f])])
% Psi: Set of set of literals (e.g. [[f,g],[i,-h]])
% EffRap: Set of literals (e.g. [f,g,-h]) 
% PreRap: Set of literals (e.g. [f,g,-h]) 
% Ram: Set of couples effect-precondition (e.g [([f,-g],[-h]),([g,i],[f])])
rulesSplit(Ra,Psi,Rap,Ram) :- 
    partition(rulesPartitionChk1(Psi),Ra,T,Ram0),
    listPartition(T,Rap,Ram1),
    p(Rap,PreRap), 
    consistentChk(PreRap),
    ord_union(Ram0,Ram1,Ram), 
    rulesPartitionChk2(Ram,PreRap).

% Collect the Literal in Obs such that the relative fluent are sensed by the sensing action Act.
% Act: Sensing action (e.g. a)
% Obs: Set of literals (e.g. [f,g,-h])
% Literal: Set of literals (e.g. [f,g,-h])
sensedLiteral(Act,Obs,Literal) :-
    senseAct(Act,F),
    !,
    unionWithNegation([F],T),
    nth0(0,Obs,Obs_0),
    ord_intersection(Obs_0,T,[Literal|_]).

% Calculate a state State containing Set.
%
% Set: Set of literals (e.g. [f,g,-h])
% State: Set of literals (e.g. [f,g,-h]) 
stateExpansion(Set,State) :-
    fluents(Set,F_Set),
    complementaryFluents(F_Set,T0),
    (
        T0 \== [] ->
        (          
            nth0(0,T0,F),
            literal(F,L),
            ord_add_element(Set,L,Set0), 
            closure(Set0,Set1),            
            stateExpansion(Set1,State)
        );
        (
            stateChk(Set),
            State = Set
        )        
    ).
    
% Check if there is some s.c.l. that makes Set inconsistent.
%
% Set: Set of literals (e.g. [f,g,-h])
staticLawsChk(Set) :- 
    negations(Set,NegatedSet),
    forall((staticLaw(Eff,Pre),ord_subset(Pre,Set)), ord_disjoint(Eff,NegatedSet)).


% Collect in Pre the preconditions of the rules in Rules.
%
% Rules: Set of couples effect-precondition (e.g [([f,-g],[-h]),([g,i],[f])])
% Pre: Set of literals (e.g. [f,g,-h])
p(Rules,Pre) :- 
    seconds(Rules,T), 
    ord_union(T,Pre).

% Check if Phi is a regression result of Psi
%
% Phi: Set of literals (e.g. [f,g,-h])
% EffRap: Set of literals (e.g. [f,g,-h])
% Psi: Set of literals (e.g. [f,g,-h])
computedRegressionResultChk(EffScl,EffRap,Psi,Phi) :- 
    negations(EffRap,NegatedEffRap),
    negations(EffScl,NegatedEffScl),
    forall(
        (
            stateExpansion(Phi,S_Phi), 
            ord_subtract(S_Phi,Phi,Gamma) 
        ),( 
            ord_subtract(Phi,NegatedEffRap,T0), 
            ord_intersection(NegatedEffScl, T0, NegatedDelta10, Delta00),
            listPartition(NegatedDelta10,Delta01,NegatedDelta1), 
            ord_union(Delta00,Delta01,Delta),
            negations(NegatedDelta1,Delta1), 
            ord_union([Delta,Gamma,EffRap],T2), 
            closure(T2,T3), 
            stateChk(T3),
            ord_subset(Psi,T3),
            ord_subset(Delta1,T3)
        )
    ).  

% Check if the knowledge doesn't decrease in the progression Phi -> Psi.
%
% Phi: Set of set of literals (e.g. [[f,g],[i,-h]])
% Psi: Set of set of literals (e.g. [[f,g],[i,-h]])
progressionChk(Phi,Psi) :-
    fluents(Phi,F_Phi), 
    fluents(Psi,F_Psi),
    ord_subset(F_Phi,F_Psi).

% Check if a rule (Eff,Pre) is compatible with Psi
%
% Psi: Set of literals (e.g. [f,g,-h])
% Eff: Set of literals (e.g. [f,g,-h])
rulesPartitionChk1(Psi,(Eff,_)) :- 
    ord_union(Psi,Eff,T), 
    consistentChk(T).

% Check if  preconditions of the rules in Ra- are satisfied by pre(Ra+)
%
% Ram: Set of couples effect-precondition (e.g [([f,-g],[-h]),([g,i],[f])])
% PreRap: Set of literals (e.g. [f,g,-h])
rulesPartitionChk2(Ram,PreRap) :- forall(member((_,Pre),Ram),\+ord_subset(Pre,PreRap)).


% Check if State is complete and consistent
%
% State: Set of literals (e.g. [f,g,-h]) 
stateChk(State) :- 
    assertChk(sortedChk(State)), 
    fluents(AllFluents), 
    length(AllFluents,N), 
    fluents(State,F_State),
    length(F_State,N), 
    forall(staticLaw(Eff,Pre), (ord_subset(Pre,State) -> ord_subset(Eff,State) ; true)).

% ================================= PROGRESSION =================================

% Calculate the progression Phi -> Psi by the History.
%
% Phi: Set of set of literals (e.g. [[f,g],[i,-h]])
% History: List of couples action-observation (e.g. [(a,[[f],[g]]),(b,[[-h]])])
% Psi: Set of set of literals (e.g. [[f,g],[i,-h]])
progression(Phi,History,Psi) :-
    findall(Phi0_i,(member(Phi_i,Phi),closure(Phi_i,Phi0_i),consistentChk(Phi0_i)),Phi0),
    foldl(progressionActionObservation,History,Phi0,Psi).

% Calculate the progression Phi -> Psi by the couple action-observation (Act,Obs).
%
% Phi:  Set of set of literals (e.g. [[f,g],[i,-h]])
% Act: Action (e.g. a)
% Obs: Set of set of literals (e.g. [[f,g],[i,-h]])
% Psi: Set of set of literals (e.g. [[f,g],[i,-h]])
progressionActionObservation((Act,Obs),Phi,Psi) :-
    assertChk(once((dynamicLaw(Act,_,_);senseAct(Act,_)))),
    forall(member(Phi_i,Phi), executableChk(Act,Phi_i)),
    (
        once(dynamicLaw(Act,_,_)) -> 
            progressionOnticAction(Act,Phi,Psi0) 
        ; 
            Psi0 = Phi
    ),
    forceObservation(Psi0,Obs,Psi1),
    minimals(Psi1,smallerBySubset,Psi2),
    sort(Psi2,Psi).

% Calculate the progression Phi -> Psi by the ontic action Act.
%
% Phi: Set of set of literals (e.g. [[f,g],[i,-h]])
% Act: Ontic action (e.g. a)
% Psi: Set of set of literals (e.g. [[f,g],[i,-h]])
progressionOnticAction(Act,Phi,Psi) :-
    findall(Psi_i,(member(Phi_i,Phi),progressionOnticActionBase(Act,Phi_i,Psi_i)),Psi0),   
    Psi0 \== [],
    sort(Psi0,Psi).

% Calculate the progression Phi -> Psi by the ontic action Act.
%
% Phi: Set of set of literals (e.g. [[f,g],[i,-h]])
% Act: Ontic action (e.g. a)
% Psi: Set of set of literals (e.g. [[f,g],[i,-h]])
progressionOnticActionBase(Act,Phi,Psi) :- 
    e(Act,Phi,Eff), 
    unionWithNegation(Eff,T0), 
    ord_subtract(Phi,T0,T1), 
    listPartition(T1,Inertia,_),
    ord_union(Inertia,Eff,Psi0), 
    closure(Psi0,Psi),  
    consistentChk(Psi),
    progressionChk(Phi,Psi).
    
% ================================= REGRESSION =================================

% Calculate Phi regression of the History.
%
% History: List of couples action-observation (e.g. [(a,[[f],[g]]),(b,[[-h]])])
% Phi: Set of set of literals (e.g. [[f,g],[i,-h]])
regression(History,Phi) :- 
    foldr(regressionActionObservation,History,[],Phi).

% Calculate the regression Phi <- Psi by the couple action-observation (Act,Obs).
%
% Psi:  Set of set of literals (e.g. [[f,g],[i,-h]])
% Act: Action (e.g. a)
% Obs: Set of set of literals (e.g. [[f,g],[i,-h]])
% Phi: Set of set of literals (e.g. [[f,g],[i,-h]])
regressionActionObservation((Act,Obs),Psi,Phi) :-
    assertChk(once((dynamicLaw(Act,_,_);senseAct(Act,_)))),
    format('INFO: Regessing: (~w)~n',[(Act,Obs)]),
    forceObservation(Psi,Obs,Psi0),
    (
        once(dynamicLaw(Act,_,_)) -> 
        (
            regressionOnticAction(Act,Psi0,Phi0)            
        );(
            sensedLiteral(Act,Obs,Literal),
            regressionSensingAction(Act,Literal,Psi0,Phi1),
            maplist(ord_union([Literal]),Phi1,Phi0)
        )
    ),
    minimals(Phi0,smallerBySubset,Phi2),
    sort(Phi2,Phi).

% Calculate the regression Phi <- Psi by the ontic action Act.
%
% Psi: Set of set of literals (e.g. [[f,g],[i,-h]])
% Act: Ontic action (e.g. a)
% Phi: Set of set of literals (e.g. [[f,g],[i,-h]])
regressionOnticAction(Act,Psi,Phi) :-
    exeCnd(Act,Eta_a),
    !, 
    findall((Eff,Pre),dynamicLaw(Act,Eff,Pre),Ra),
    findall(Phi_i,(member(Psi_i,Psi),regressionOnticActionBase(Eta_a,Ra,Psi_i,Phi_i)),Phi0),
    Phi0 \== [],
    sort(Phi0,Phi).

% Calculate the regression Phi <- Psi given Eta_a and Ra, the executability conditions and the rules of A.
%
% Psi: Set of literals (e.g. [f,g,-h])
% Eta_a: Set of literals (e.g. [f,g,-h])
% Ra: Set of couples effect-precondition (e.g [([f,-g],[-h]),([g,i],[f])])
% Phi: Set of literals (e.g. [f,g,-h])
regressionOnticActionBase(Eta_a,Ra,Psi,Phi) :-
    minimalPotentialRegressionResult(Eta_a,Ra,Psi,Phi0,EffRap),
    closure(EffRap,ClosedEffRap),
    negations(ClosedEffRap,NegatedClosedEffRap),
    findall(Eff,(staticLaw(Eff,Pre),ord_disjoint(Eff,NegatedClosedEffRap),ord_disjoint(Pre,NegatedClosedEffRap)),T0),
    ord_union(T0,EffScl),
    minimalComputedRegressionResult(EffScl,EffRap,Psi,Phi0,Phi).

% Calculate a minimal (w.r.t. ⊆) potential regression result MinPhi.
%
% Psi: Set of literals (e.g. [f,g,-h])
% Eta_a: Set of literals (e.g. [f,g,-h])
% MinPhi: Set of literals (e.g. [f,g,-h])
% EffRap: Set of literals (e.g. [f,g,-h])   
minimalPotentialRegressionResult(Eta_a,Ra,Psi,Phi,EffRap) :-
    rulesSplit(Ra,Psi,Rap,Ram), 
    p(Rap,PreRap),
    e(Rap,EffRap),
    ord_union(Eta_a,PreRap,Phi0),
    closure(Phi0,Phi1),
    blockingExpansion(Phi1,Ram,Phi2),
    closure(Phi2,Phi),
    consistentChk(Phi).   

% Calculate the minimal (w.r.t. ⊆) regression result Psi superset of Phi0.
%
% Psi: Set of literals (e.g. [f,g,-h])
% EffRap: Set of literals (e.g. [f,g,-h])    
% Phi0: Set of literals (e.g. [f,g,-h])
% Phi: Set of literals (e.g. [f,g,-h])   
minimalComputedRegressionResult(EffScl,EffRap,Psi,Phi0,Phi) :- 
    (
        \+computedRegressionResultChk(EffScl,EffRap,Psi,Phi0) ->
        (
            fluents(Phi0,T0),
            complementaryFluents(T0,[F|_]),            
            literal(F,L), 
            ord_add_element(Phi0,L,Phi1),
            closure(Phi1,Phi2),
            consistentChk(Phi2),
            ord_subtract(Phi2,Phi1,T1),
            ord_del_element(T1,L,T2),
            ord_union(T2,EffScl,EffScl0),
            minimalComputedRegressionResult(EffScl0,EffRap,Psi,Phi2,Phi) 
        );(   
            Phi = Phi0
        )
    ).     

% Calculate the regression Phi <- Psi by the sensing action A.
%
% Psi: Set of set of literals (e.g. [[f,g],[i,-h]])
% Act: Sensing action (e.g. a)
% Phi: Set of set of literals (e.g. [[f,g],[i,-h]])
% SensedValue: Set of literals (e.g. [f,g,-h])   
regressionSensingAction(Act,SensedValue,Psi,Phi) :-
    exeCnd(Act,Eta_a), 
    !,
    findall(Phi_i,(member(Psi_i,Psi),regressionSensingActionBase(Eta_a,SensedValue,Psi_i,Phi_i)),Phi0),
    Phi0 \== [],
    sort(Phi0,Phi).

% Calculate the regression Phi <- Psi by the sensing action A.
%
% Psi: Set of set of literals (e.g. [[f,g],[i,-h]])
% Act: Sensing action (e.g. a)
% Phi: Set of set of literals (e.g. [[f,g],[i,-h]])
% SensedValue: Set of literals (e.g. [f,g,-h])  
regressionSensingActionBase(Eta_a,SensedValue,Psi,Phi) :-
    unionWithNegation([SensedValue],ToDel),
    ord_subtract(Psi,ToDel,Phi0), 
    ord_union(Eta_a,Phi0,Phi1),
    closure(Phi1,Phi),
    ord_add_element(Phi,SensedValue,T),
    consistentChk(T).
    
    
% ================================= REVISION =================================


revisionBySatoh(Psi,Phi,Revision) :- revision(Psi,Phi,diffBySatoh,smallerBySatoh,Revision).

revisionByDalal(Psi,Phi,Revision) :- revision(Psi,Phi,diffByDalal,smallerByDalal,Revision).

revision(Psi,Phi,DiffPre,SmallerPre,Revision) :-    
    findall(S_Psi,(member(Psi_i,Psi),stateExpansion(Psi_i,S_Psi)),SigmaPsi0),
    sort(SigmaPsi0,SigmaPsi),
    findall((Diff,Phi_i), (member(Phi_i,Phi),stateExpansion(Phi_i,S_Phi),member(S_Psi,SigmaPsi),call(DiffPre,S_Psi,S_Phi,Diff)), Revision0),
    minimals(Revision0,SmallerPre,T),
    seconds(T,Minimals0),
    sort(Minimals0,Revision).

diffBySatoh(State1,State2,Diff) :- 
    ord_symdiff(State1,State2,Diff).
    
diffByDalal(State1,State2,Diff) :- 
    ord_symdiff(State1,State2,T),
    length(T,Diff).    

smallerBySatoh(Couple1,Couple2) :- 
    Couple1 = (Diff1,_),
    Couple2 = (Diff2,_), 
    Diff1 \== Diff2, 
    ord_subset(Diff1,Diff2).

smallerByDalal(Couple1,Couple2) :- 
    Couple1 = (Diff1,_),
    Couple2 = (Diff2,_), 
    Diff1 \== Diff2, 
    Diff1 < Diff2.
