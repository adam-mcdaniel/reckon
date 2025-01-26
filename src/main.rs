
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

fn eval<S>(mut input: &str, env: &mut Env<S>) -> Result<(), String> where S: Solver {


    while !input.trim().is_empty() {

        match parse_rule(input) {
            Ok((rest, rule)) => {
                env.add_rule(rule);
                input = rest;
            }
            Err(rule_error) => {
                match parse_query(input) {
                    Ok((rest, query)) => {
                        input = rest;
                        let (solutions, duration) = time_it(|| {
                            env.find_solutions(&query)
                        });
                        match solutions {
                            Ok(solutions) => {
                                info!("Found {} solutions for {} in {:?}", solutions.len(), query, duration);
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
                            },
                            Err(terms) => {
                                error!("Could not find solution for query: ");
                                for term in terms {
                                    error!("{}", term);
                                }
                            }
                        }
                    }
                    Err(_) => {
                        // Parse the search config
                        match env.search_config_mut().parse(input) {
                            Ok(rest) => {
                                input = rest;
                            }
                            Err(_) => {
                                error!("Syntax error:");
                                let error_string = match rule_error {
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

fn main() {
    // Set up logging with the `tracing` crate, with debug level logging.
    let _ = tracing_subscriber::fmt::SubscriberBuilder::default()
        .with_max_level(tracing::Level::INFO)
        .init();
    

    let mut env = Env::<DefaultSolver>::new(&[]);

    eval(r#"
        options(
            depth_limit=1000,
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

        options(solution_limit=1).

        ?- ~add(A, A, s(s(s(s(0))))), add(A, B, C), leq(A, s(s(s(0)))), leq(B, s(s(s(0)))), leq(C, s(s(s(s(0))))).


        ?- square(X, s(s(0))).
        ?- square(X, s(s(s(s(0))))).
    "#, &mut env).unwrap();

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