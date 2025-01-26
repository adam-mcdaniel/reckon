
% "is_nat(s(X)) :- is_nat(X).".parse().unwrap(),
% "is_nat(0).".parse().unwrap(),
% "add(X, s(Y), s(Z)) :- add(X, Y, Z).".parse().unwrap(),
% "add(X, 0, X) :- is_nat(X).".parse().unwrap(),
% "leq(0, X) :- is_nat(X).".parse().unwrap(),
% "leq(s(X), s(Y)) :- leq(X, Y).".parse().unwrap(),

% "geq(X, Y) :- leq(Y, X).".parse().unwrap(),
% "eq(X, Y) :- leq(X, Y), leq(Y, X).".parse().unwrap(),
% "neq(X, Y) :- ~eq(X, Y).".parse().unwrap(),

% "lt(X, Y) :- leq(X, Y), ~eq(X, Y).".parse().unwrap(),
% "gt(X, Y) :- geq(X, Y), ~eq(X, Y).".parse().unwrap(),


% "mul(X, s(Y), Z) :- mul(X, Y, W), add(X, W, Z).".parse().unwrap(),
% "mul(X, 0, 0) :- is_nat(X).".parse().unwrap(),
% "square(X, Y) :- mul(X, X, Y).".parse().unwrap(),


is_nat(s(X)) :- is_nat(X).
is_nat(0).
add(X, s(Y), s(Z)) :- add(X, Y, Z).
add(X, 0, X) :- is_nat(X).

leq(0, X) :- is_nat(X).
leq(s(X), s(Y)) :- leq(X, Y).

geq(X, Y) :- leq(Y, X).
eq(X, Y) :- leq(X, Y), leq(Y, X).
neq(X, Y) :- ~eq(X, Y).

lt(X, Y) :- leq(X, Y), ~eq(X, Y).
gt(X, Y) :- geq(X, Y), ~eq(X, Y).

mul(X, s(Y), Z) :- mul(X, Y, W), add(X, W, Z).
mul(X, 0, 0) :- is_nat(X).

square(X, Y) :- mul(X, X, Y).

?- square(s(s(0)), X), options