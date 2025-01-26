use crate::*;
use tracing::{info, error};

/// Time a function and return the result and the duration
pub fn time_it<F: FnMut() -> R, R>(mut f: F) -> (R, std::time::Duration) {
    let start = std::time::Instant::now();
    let result = f();
    let duration = start.elapsed();
    (result, duration)
}

/// Convert an integer to a Peano number
pub fn int_to_peano(n: usize) -> String {
    let n = n;
    let mut peano = String::from("0");
    for _ in 0..n {
        peano = format!("s({})", peano);
    }
    peano
}

/// Convert a Peano number to an integer
pub fn peano_to_int(peano: &str) -> Option<usize> {
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


/// Evaluate a string of rules, queries, and search configuration settings
/// in an environment with a given solver
pub fn eval<S>(input: &str, env: &mut Env<S>) -> Result<(), String> where S: Solver {
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
                                let mut error_string = "Could not find solution for query:\n".to_string();
                                // error!("Could not find solution for query: ");
                                for term in terms {
                                    error_string.push_str(&format!("{} ", term));
                                }

                                return Err(error_string);
                            }
                        }
                    }
                    Err(query_error) => {
                        // Parse the search config
                        match env.search_config_mut().parse(input) {
                            Ok(rest) => {
                                env.reset_solver();
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
                                };

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