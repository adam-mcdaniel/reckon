
use logic::*;
use logic::solvers::*;
use tracing::*;

fn time_it<F: FnMut() -> R, R>(mut f: F) -> (R, std::time::Duration) {
    let start = std::time::Instant::now();
    let result = f();
    let duration = start.elapsed();
    (result, duration)
}

fn int_to_peano(n: usize) -> String {
    let mut n = n;
    let mut peano = String::from("0");
    for _ in 0..n {
        peano = format!("s({})", peano);
    }
    peano
}

fn peano_to_int(peano: &str) -> Option<usize> {
    let mut n = 0;
    let mut peano = peano;
    while peano.starts_with("s(") {
        n += 1;
        peano = &peano[2..peano.len() - 1];
    }
    if peano == "0" {
        Some(n)
    } else {
        None
    }
}

fn eval<S>(input: &str, env: &mut Env<S>) -> Result<(), String> where S: Solver {
    let input = strip_cpp_like_comments(input);
    eval_helper(&input, env)
}

fn eval_helper<S>(mut input: &str, env: &mut Env<S>) -> Result<(), String> where S: Solver {
    while !input.trim().is_empty() {
        input = input.trim();
        match parse_rule(input) {
            Ok((rest, rule)) => {
                env.add_rule(rule);
                input = rest;
            }
            Err(rule_error) => {
                match parse_query(input) {
                    Ok((rest, query)) => {
                        env.reset_steps();
                        input = rest;
                        let (solutions, duration) = time_it(|| {
                            env.find_solutions(&query)
                        });
                        match solutions {
                            Ok(solutions) => {
                                info!("{}", "=".repeat(60));
                                info!("Found {} solutions for {} ({} steps) in {:?}", solutions.len(), query, env.get_steps(), duration);
                                for (i, solution) in solutions.iter().enumerate() {
                                    info!("Solution #{}: ", i + 1);
                                    for (var, term) in solution.var_bindings() {
                                        let peano = peano_to_int(&term.to_string());
                                        match peano {
                                            Some(n) => {
                                                info!("{} = (peano for {}) {}", var, n, term);
                                            },
                                            None => {
                                                info!("{} = {}", var, term);
                                            }
                                        }
                                    }
                                }
                                info!("{}", "=".repeat(60));
                                info!("");
                            },
                            Err(terms) => {
                                error!("Could not find solution for query: ");
                                for term in terms {
                                    error!("{}", term);
                                }
                            }
                        }
                    }
                    Err(query_error) => {
                        // Parse the search config
                        match env.search_config_mut().parse(input) {
                            Ok(rest) => {
                                input = rest;
                            }
                            Err(_) => {
                                error!("Syntax error:");

                                let error = match input.chars().next() {
                                    Some('?') => query_error,
                                    _ => rule_error
                                };

                                let error_string = match error {
                                    nom::Err::Error(e) | nom::Err::Failure(e) => nom::error::convert_error(input, e),
                                    nom::Err::Incomplete(_) => "Incomplete input".to_string(),
                                    _ => "Unknown error".to_string()
                                };
                                for line in error_string.lines() {
                                    error!("{}", line);
                                }

                                return Err(error_string);
                            }
                        }
                    }
                }
            }
        }

    }

    Ok(())
}

fn strip_cpp_like_comments(input: &str) -> String {
    let mut output = String::new();
    let mut input_chars = input.chars().peekable();
    while let Some(c) = input_chars.next() {
        if c == '/' {
            if let Some('/') = input_chars.peek() {
                // Skip the rest of the line
                while let Some(c) = input_chars.next() {
                    if c == '\n' {
                        break;
                    }
                }
            } else if let Some('*') = input_chars.peek() {
                // Skip the block comment
                while let Some(c) = input_chars.next() {
                    if c == '*' {
                        if let Some('/') = input_chars.peek() {
                            input_chars.next();
                            break;
                        }
                    }
                }
            } else {
                output.push(c);
            }
        } else if c == '%' {
            // Skip the rest of the line
            while let Some(c) = input_chars.next() {
                if c == '\n' {
                    break;
                }
            }
        } else {
            output.push(c);
        }
    }
    output
}

fn main() {
    // Set up logging with the `tracing` crate, with debug level logging.
    let _ = tracing_subscriber::fmt::SubscriberBuilder::default()
        .with_max_level(tracing::Level::INFO)
        .init();
    

    let mut env = Env::<MemoizingSolver>::new(&[])
    .with_search_config(&SearchConfig::default().with_sorter(100, |_, query: &Query| query.size()));

    // eval(r#"
    //     options(
    //         depth_limit=3000,
    //         traversal="breadth_first",
    //         pruning=false,
    //         require_rule_head_match=true,
    //         reduce_query=false,
    //         solution_limit=5
    //     ).

    //     // Define the natural numbers
    //     // 
    //     // NOTE: s(X) means the successor of X
    //     is_nat(s(X)) :- is_nat(X). // <-- s(X) is a natural number if X is a natural number
    //     is_nat(0).                 // <-- 0 is a natural number



    //     // Define addition of natural numbers
    //     add(X, 0, X) :- is_nat(X).          // <-- X + 0 = X
    //     add(X, s(Y), s(Z)) :- add(X, Y, Z). // <-- X + s(Y) = s(X + Y)


    //     // Query: What are X and Y such that X + Y = 3?
    //     ?- add(X, Y, s(s(s(0)))). // <-- Find X and Y such that X + Y = 3


    //     // Define multiplication of natural numbers
    //     mul(X, 0, 0) :- is_nat(X).
    //     mul(X, s(Y), Z) :- mul(X, Y, W), add(X, W, Z).

    //     // Query: What are X and Y such that X + Y = 4, and X * Y != 4?
    //     ?- mul(X, Y, s(s(s(s(0))))). // <-- Find X and Y such that X + Y = 4, and X * Y != 4

    //     options(solution_limit=1).

    //     // Define less-than-or-equal-to relation on natural numbers
    //     leq(0, X) :- is_nat(X).
    //     leq(s(X), s(Y)) :- leq(X, Y).

    //     // Query: What are A, B, and C such that A + B 
    //     ?- add(A, B, C), ~add(A, A, s(s(s(s(0))))), leq(A, s(s(s(0)))), leq(B, s(s(s(0)))), leq(C, s(s(s(s(0))))).
    // "#, &mut env).unwrap();
    /*
    eval(r#"
        options(
            depth_limit=3000,
            traversal="breadth_first",
            pruning=false,
            require_rule_head_match=true,
            reduce_query=false,
            solution_limit=5
        ).


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

        ?- add(A, B, s(s(s(s(0))))).


        ?- ~add(A, A, s(s(s(s(0))))), add(A, B, C), leq(A, s(s(s(0)))), leq(B, s(s(s(0)))), leq(C, s(s(s(s(0))))).

        options(solution_limit=1).

        ?- square(X, s(s(0))).

        ?- square(s(s(0)), X), square(X, Y).
    "#, &mut env).unwrap();
     */

    // Now for some *interesting* programs
    // Write a system F type checker
eval(r#"
options(
    depth_limit=50,
    width_limit=5,
    traversal="breadth_first",
    pruning=false,
    require_rule_head_match=true,
    reduce_query=false,
    solution_limit=1,
    clean_memoization=true
).

term(var(X)) :- atom(X).

atom(X).

% Lambda abstraction
term(abs(X, T, Body)) :- atom(X), type(T), term(Body).

% Application
term(app(Func, Arg)) :- term(Func), term(Arg).

% Type abstraction (universal quantification: ΛX. T)
term(tabs(TVar, Body)) :- atom(TVar), term(Body).

% Type application (specializing a polymorphic type: T [τ])
term(tapp(Func, T)) :- term(Func), type(T).

% Base types
type(base(T)) :- atom(T). % Example: `int`, `bool`

% Arrow types (functions)
type(arrow(T1, T2)) :- type(T1), type(T2). % Example: T1 -> T2

% Universal quantifiers (∀X. T)
type(forall(TVar, T)) :- atom(TVar), type(T).

bind(X, T) :- atom(X), type(T).

context(nil).
context([]).
context([bind(X, T) | Rest]) :- atom(X), type(T), context(Rest).

member(X, [X | _]).
member(X, [_ | Rest]) :- member(X, Rest).


has_type(Ctx, var(X), T) :-
    member(bind(X, T), Ctx).

has_type(Ctx, abs(X, T, Body), arrow(T, TBody)) :-
    has_type([bind(X, T) | Ctx], Body, TBody).

has_type(Ctx, app(Func, Arg), T2) :-
    has_type(Ctx, Func, arrow(T1, T2)),
    has_type(Ctx, Arg, T1).

has_type(Ctx, tabs(TVar, Body), forall(TVar, TBody)) :-
    has_type(Ctx, Body, TBody).

has_type(Ctx, tapp(Func, Type), TSubstituted) :-
    has_type(Ctx, Func, forall(TVar, TBody)),
    substitute(TBody, TVar, Type, TSubstituted).

eq(T1, T1).
eq(base(T1), base(T2)) :- eq(T1, T2).
eq(arrow(T1, T2), arrow(T3, T4)) :- eq(T1, T3), eq(T2, T4).
eq(forall(X, T1), forall(X, T2)) :-
    eq(T1, T2). % Bodies of the quantified types must be equal
neq(T1, T2) :- ~eq(T1, T2).

% Substitution base case: If the type is the type variable being substituted, replace it.
substitute(base(T), TVar, Replacement, Replacement) :-
    eq(T, TVar).

% If the type is not the variable being replaced, leave it unchanged.
substitute(base(T), TVar, _, base(T)) :-
    eq(T, TVar).

% For arrow types, substitute in both the domain and codomain.
substitute(arrow(T1, T2), TVar, Replacement, arrow(T1Sub, T2Sub)) :-
    substitute(T1, TVar, Replacement, T1Sub),
    substitute(T2, TVar, Replacement, T2Sub).

% For universal quantifiers, substitute in the body only if the bound variable is not the same.
substitute(forall(TVarInner, TBody), TVar, Replacement, forall(TVarInner, TBodySub)) :-
    neq(TVar, TVarInner), % Avoid variable capture
    substitute(TBody, TVar, Replacement, TBodySub).

?- has_type([], abs(x, A, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, B, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, C, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, D, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, E, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, F, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, G, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, H, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, A, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, B, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, C, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, D, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, E, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, F, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, G, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, H, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, A, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, B, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, C, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, D, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, E, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, F, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, G, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, H, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, A, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, B, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, C, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, D, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, E, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, F, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, G, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, H, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, A, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, B, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, C, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, D, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, E, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, F, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, G, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, H, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, A, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, B, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, C, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, D, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, E, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, F, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, G, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, H, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, A, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, B, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, C, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, D, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, E, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, F, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, G, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, H, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, A, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, B, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, C, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, D, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, E, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, F, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, G, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, H, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, A, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, B, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, C, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, D, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, E, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, F, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, G, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, H, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, A, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, B, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, C, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, D, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, E, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, F, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, G, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, H, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, A, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, B, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, C, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, D, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, E, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, F, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, G, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, H, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, A, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, B, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, C, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, D, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, E, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, F, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, G, abs(y, base(float), var(x))), T).
?- has_type([], abs(x, H, abs(y, base(float), var(x))), T).
"#, &mut env).unwrap();

    /*
?- context([bind(var(x), base(float)) | [bind(var(y), base(int))]]).
?- member(bind(var(x), base(float)), [bind(var(x), base(float)) | [bind(var(y), base(int))]]).
?- has_type([bind(x, base(float)) | [bind(y, base(int))]], var(x), T).

    let rules: Vec<Rule> = vec![
        "is_nat(s(X)) :- is_nat(X).".parse().unwrap(),
        "is_nat(0).".parse().unwrap(),
        "add(X, s(Y), s(Z)) :- add(X, Y, Z).".parse().unwrap(),
        "add(X, 0, X) :- is_nat(X).".parse().unwrap(),
        "leq(0, X) :- is_nat(X).".parse().unwrap(),
        "leq(s(X), s(Y)) :- leq(X, Y).".parse().unwrap(),
        
        "geq(X, Y) :- leq(Y, X).".parse().unwrap(),
        "eq(X, Y) :- leq(X, Y), leq(Y, X).".parse().unwrap(),
        "neq(X, Y) :- ~eq(X, Y).".parse().unwrap(),

        "lt(X, Y) :- leq(X, Y), ~eq(X, Y).".parse().unwrap(),
        "gt(X, Y) :- geq(X, Y), ~eq(X, Y).".parse().unwrap(),
        
        
        "mul(X, s(Y), Z) :- mul(X, Y, W), add(X, W, Z).".parse().unwrap(),
        "mul(X, 0, 0) :- is_nat(X).".parse().unwrap(),
        "square(X, Y) :- mul(X, X, Y).".parse().unwrap(),
    ];


    let query: Query = "?- ~add(A, A, s(s(s(s(0))))), add(A, B, C), leq(A, s(s(s(0)))), leq(B, s(s(s(0)))), leq(C, s(s(s(s(0))))).".parse().unwrap();

    let mut env = Env::<DefaultSolver>::new(&rules);

    let solution = env.prove_true(&query).unwrap();

    println!("{}", solution);
     */



    /*
    // Set up logging with the `tracing` crate, with debug level logging.
    let _ = tracing_subscriber::fmt::SubscriberBuilder::default()
        .with_max_level(tracing::Level::INFO)
        .init();
    
    let rules: Vec<Rule> = vec![
        "is_nat(s(X)) :- is_nat(X).".parse().unwrap(),
        "is_nat(0).".parse().unwrap(),
        "add(X, s(Y), s(Z)) :- add(X, Y, Z).".parse().unwrap(),
        "add(X, 0, X) :- is_nat(X).".parse().unwrap(),
        "leq(0, X) :- is_nat(X).".parse().unwrap(),
        "leq(s(X), s(Y)) :- leq(X, Y).".parse().unwrap(),
        
        "geq(X, Y) :- leq(Y, X).".parse().unwrap(),
        "eq(X, Y) :- leq(X, Y), leq(Y, X).".parse().unwrap(),
        "neq(X, Y) :- ~eq(X, Y).".parse().unwrap(),

        "lt(X, Y) :- leq(X, Y), ~eq(X, Y).".parse().unwrap(),
        "gt(X, Y) :- geq(X, Y), ~eq(X, Y).".parse().unwrap(),
        
        
        "mul(X, s(Y), Z) :- mul(X, Y, W), add(X, W, Z).".parse().unwrap(),
        "mul(X, 0, 0) :- is_nat(X).".parse().unwrap(),
        "square(X, Y) :- mul(X, X, Y).".parse().unwrap(),
        // "isprime(X, Y, Z) :- is_nat(X), is_nat(Y), is_nat(Z), ~eq(X, s(0)), ~eq(Y, s(0)),  ~eq(X, 0), ~eq(Y, 0), ~eq(Z, 0), ~mul(X, Y, Z).".parse().unwrap(),
    ];

    // let query: Query = "?- add(A, B, s(s(s(s(0))))).".parse().unwrap();
    let mut n = String::from("0");
    for _ in 0..16 {
        n = format!("s({})", n);
    }

    let query: Query = format!("?- ~add(A, A, s(s(s(s(0))))), add(A, B, C), leq(A, s(s(s(0)))), leq(B, s(s(s(0)))), leq(C, s(s(s(s(0))))).").parse().unwrap();
    // let mut query: Query = format!(r#"?- neq(A, s(0)), mul(A, B, {n})."#).parse().unwrap();
    // let mut query: Query = format!(r#"?- lt(A, {n})."#).parse().unwrap();
        
    // let query: Query = "?- neq(A, s(0)), neq(B, s(0)), mul(A, B, s(s(s(s(0))))).".parse().unwrap();
    // let query: Query = "?- isprime(X, Y, s(s(s(0)))).".parse().unwrap();

    // &SearchConfig::default()
    //         // .with_step_limit(1000)
    //         .with_traversal(Traversal::BreadthFirst)
    //         // .with_traversal(Traversal::DepthFirst)
    //         .with_depth_limit(3000)
    //         .with_width_limit(5)
    //         .with_pruning(false)
    //         // .with_sorter(100, |_, query: &Query| query.size())
    //         // .with_sorter(100, |_, query: &Query| usize::MAX - query.size())
    //         .with_require_rule_head_match(true)
    //         .with_reduce_query(false)
    //         .with_solution_limit(5)

    let mut config = SearchConfig::default();
    config.parse(r#"
        options(
            depth_limit=1000,
            traversal="breadth_first",
            pruning=false,
            require_rule_head_match=true,
            reduce_query=false,
            solution_limit=5
        ).
    "#).unwrap();

    let mut env = Env::<DefaultSolver>::new(&rules)
        .with_search_config(&config);

    // println!("{}", query);
    // println!("{}", env.prove_true(&mut query).unwrap());
    // return;

    // match env.prove_true(&query) {
    //     Ok(solution) => {
    //         println!("{}", solution);
    //     }
    //     Err(terms) => {
    //         error!("Could not find solution for query: ");
    //         for term in terms {
    //             error!("{}", term);
    //         }
    //     }
    // }
    let (solutions, duration) = time_it(|| {
        env.find_solutions(&query)
    });
    // let solutions = env.find_solutions(&query, 5);

    match solutions {
        Ok(solutions) => {
            info!("Found {} solutions in {:?}", solutions.len(), duration);
            for (i, solution) in solutions.iter().enumerate() {
                info!("Solution #{}: ", i + 1);
                for (var, term) in solution.var_bindings() {
                    let peano = peano_to_int(&term.to_string());
                    match peano {
                        Some(n) => {
                            info!("{} = (peano for {}) {}", var, n, term);
                        },
                        None => {
                            info!("{} = {}", var, term);
                        }
                    }
                    // Print the integer value of the Peano number
                    

                }
            }
        },
        Err(terms) => {
            error!("Could not find solution for query: ");
            for term in terms {
                error!("{}", term);
            }
        }
    }

    */
}