
use reckon::*;
use clap::Parser;
use std::fmt::{Display, Write};
use std::fs::read_to_string;
use tracing::{info, error};

// The stack sizes of the threads used to compile the code.
const RELEASE_STACK_SIZE_MB: usize = 512;
const DEBUG_STACK_SIZE_MB: usize = RELEASE_STACK_SIZE_MB;

/// The argument parser for the CLI.
#[derive(Parser, Debug)]
#[command(author, version, about = Some(COLORED_LOGO_1), long_about = Some(COLORED_LOGO_1), max_term_width=90)]
struct Args {
    /// The input file to read the program from. For a REPL session, omit this argument.
    #[clap(value_parser)]
    input: Option<String>,

    /// The output file to write the results to. For a REPL session, omit this argument.
    /// If this argument is omitted, the results will be printed to the console.
    #[clap(short, long)]
    output: Option<String>,
}

fn repl(mut env: Env<impl Solver>) {
    use rustyline::error::ReadlineError;
    clear_screen();
    println!("Welcome to...\n{}\n\nUse the REPL to enter your rules, queries, and search options.\nFor help, input \":help\".", COLORED_LOGO_1);
    let mut envs = Vec::new();

    let default_entry = rgb_text(">>> ", (0, 255, 0));
    let continuation_entry = rgb_text("... ", (0, 0, 255));
    if let Ok(mut rl) = rustyline::DefaultEditor::new() {
        rl.load_history(".reckon_history").ok();
        let mut current_input = String::new();
        let mut current_prompt = default_entry.clone();
        loop {
            let readline = rl.readline(&current_prompt);
            if rl.save_history(".reckon_history").is_err() {
                error!("Could not save history");
            }
            match readline {
                Ok(line) => {
                    if line.trim().chars().next() == Some(':') {
                        let line = line.trim();
                        if line == ":q" || line == ":quit" {
                            break;
                        } else if line == ":c" || line == ":clear" {
                            env.reset();
                            info!("Environment cleared.");
                            continue;
                        } else if line == ":s" || line == ":save" {
                            envs.push(env.clone());
                            info!("Environment saved.");
                            continue;
                        } else if line == ":r" || line == ":restore" {
                            if let Some(saved_env) = envs.pop() {
                                info!("Environment restored.");
                                env = saved_env;
                            } else {
                                error!("No environment to restore.");
                            }
                            continue;
                        } else if line == ":x" || line == ":examine" {
                            println!("{}", env);
                            continue;
                        } else if line == ":i" || line == ":import" {
                            // Ask for a filename
                            let filename = rl.readline("Enter the filename to import: ");
                            if let Ok(filename) = filename {
                                if let Ok(program) = read_to_string(&filename) {
                                    current_input.push_str(&program);
                                    match eval(&current_input, &mut env) {
                                        Ok(_) => {
                                            info!("OK.");
                                        },
                                        Err(error) => {
                                            for line in error.lines() {
                                                error!("{}", line);
                                            }
                                        }
                                    }
                                    current_input.clear();
                                    current_prompt = default_entry.clone();
                                    continue;
                                }
                            }

                            continue;
                        } else if line == ":h" || line == ":help" {
                            println!("Commands:");
                            println!("  :q, :quit    - Quit the REPL");
                            println!("  :i, :import  - Import a file");
                            println!("  :s, :save    - Push the current environment to be restored later");
                            println!("  :r, :restore - Pop the last environment pushed");
                            println!("  :x, :examine - Print the environment");
                            println!("  :c, :clear   - Reset the environment");
                            println!("  :h, :help    - Show this help message");
                            continue;
                        }
                    }


                    let _ = rl.add_history_entry(line.as_str());

                    if line.trim().is_empty() {
                        continue;
                    }

                    if line.trim().chars().last() == Some('.') {
                        current_input.push_str(&line);
                        match eval(&current_input, &mut env) {
                            Ok(_) => {
                                info!("OK.");
                            },
                            Err(error) => {
                                for line in error.lines() {
                                    error!("{}", line);
                                }
                            }
                        }
                        current_input.clear();
                        current_prompt = default_entry.clone();
                    } else {
                        current_input.push_str(&line);
                        current_input.push_str("\n");
                        current_prompt = continuation_entry.clone();
                    }
                },
                Err(ReadlineError::Interrupted) => {
                    current_input.clear();
                    current_prompt = default_entry.clone();
                },
                Err(ReadlineError::Eof) => {
                    // Save the history
                    let _ = rl.save_history(".reckon_history");
                    break
                },
                Err(_) => {
                    error!("Could not read input");
                    break
                }
            }
        }
    } else {
        error!("Could not initialize readline");
    }
}

fn cli() {
    let args = Args::parse();

    let mut env = Env::<MemoizingSolver>::new(&[])
        .with_search_config(&SearchConfig::default().with_sorter(100, |_, query: &Query| query.size()));
    
    match args.input {
        Some(filename) => {
            if let Ok(program) = read_to_string(&filename) {
                match eval(&program, &mut env) {
                    Ok(_) => {
                        info!("Program evaluation completed successfully.");
                    },
                    Err(error) => {
                        for line in error.lines() {
                            error!("{}", line);
                        }
                    }
                }
            } else {
                error!("Could not read file: {}", filename);
            }
        }
        None => {
            // REPL
            repl(env);
        }
    }
    // println!("{:?}", args);
}

#[allow(dead_code)]
fn print_rainbow_text(text: &str) {
    let colors = [
        "\x1b[31m", // Red
        "\x1b[33m", // Yellow
        "\x1b[32m", // Green
        "\x1b[36m", // Cyan
        "\x1b[34m", // Blue
        "\x1b[35m", // Magenta
    ];

    let mut color_index = 0;
    let mut rainbow_text = String::new();

    for ch in text.chars() {
        if ch.is_whitespace() {
            write!(rainbow_text, "{}", ch).unwrap(); // Preserve whitespace
        } else {
            write!(rainbow_text, "{}{}", colors[color_index], ch).unwrap();
            color_index = (color_index + 1) % colors.len();
        }
    }

    rainbow_text.push_str("\x1b[0m"); // Reset color
    println!("{}", rainbow_text);
}

#[allow(dead_code)]
fn print_rainbow_text_lines(text: &str) {
    let colors = [
        "\x1b[31m", // Red
        "\x1b[33m", // Yellow
        "\x1b[32m", // Green
        "\x1b[36m", // Cyan
        "\x1b[34m", // Blue
        "\x1b[35m", // Magenta,
    ];

    let mut color_index = 2;
    let mut rainbow_text = String::new();

    for ch in text.chars() {
        if ch == '\n' {
            color_index = (color_index + 1) % colors.len();
        }
        write!(rainbow_text, "{}{}", colors[color_index], ch).unwrap();
    }

    rainbow_text.push_str("\x1b[0m"); // Reset color
    println!("{}", rainbow_text);
}

#[allow(dead_code)]
fn ansi_rgb(r: u8, g: u8, b: u8) -> String {
    format!("\x1b[38;2;{};{};{}m", r, g, b)
}

#[allow(dead_code)]
fn ansi_hsv(h: f64, s: f64, v: f64) -> String {
    let (r, g, b) = hsv_to_rgb((h, s, v));
    ansi_rgb(r, g, b)
}

#[allow(dead_code)]
fn ansi_reset() -> String {
    "\x1b[0m".to_string()
}


#[allow(dead_code)]
fn hsv_to_rgb(c: (f64, f64, f64)) -> (u8, u8, u8) {
    let (h, s, v) = c;

    let c = v * s; // Chroma
    let h_prime = h / 60.0; // Sector of the color wheel
    let x = c * (1.0 - ((h_prime % 2.0) - 1.0).abs());
    let m = v - c;

    let (r, g, b) = if (0.0..1.0).contains(&h_prime) {
        (c, x, 0.0)
    } else if (1.0..2.0).contains(&h_prime) {
        (x, c, 0.0)
    } else if (2.0..3.0).contains(&h_prime) {
        (0.0, c, x)
    } else if (3.0..4.0).contains(&h_prime) {
        (0.0, x, c)
    } else if (4.0..5.0).contains(&h_prime) {
        (x, 0.0, c)
    } else if (5.0..6.0).contains(&h_prime) {
        (c, 0.0, x)
    } else {
        (0.0, 0.0, 0.0) // Fallback for invalid hue
    };

    let r = ((r + m) * 255.0).round() as u8;
    let g = ((g + m) * 255.0).round() as u8;
    let b = ((b + m) * 255.0).round() as u8;

    (r, g, b)
}

#[allow(dead_code)]
fn rgb_to_hsv(c: (u8, u8, u8)) -> (f64, f64, f64) {
    let (r, g, b) = c;
    let r = r as f64 / 255.0;
    let g = g as f64 / 255.0;
    let b = b as f64 / 255.0;

    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    // Calculate hue
    let h = if delta == 0.0 {
        0.0
    } else if max == r {
        60.0 * (((g - b) / delta) % 6.0)
    } else if max == g {
        60.0 * (((b - r) / delta) + 2.0)
    } else {
        60.0 * (((r - g) / delta) + 4.0)
    };

    let h = if h < 0.0 { h + 360.0 } else { h };

    // Calculate saturation
    let s = if max == 0.0 { 0.0 } else { delta / max };

    // Calculate value
    let v = max;

    (h, s, v)
}


#[allow(dead_code)]
fn gradient(text: impl Display, color1: (f64, f64, f64), color2: (f64, f64, f64)) -> String {
    let mut result = String::new();
    let text = text.to_string();
    

    let (mut hue, mut saturation, mut value) = color1;
    let char_count = text.chars().filter(|c| !c.is_whitespace()).count();
    // let start_hue = color1.0.min(color2.0);
    // let end_hue = color1.0.max(color2.0);
    let start_hue = color1.0;
    let end_hue = color2.0;
    let hue_step = (end_hue - start_hue) / char_count as f64;

    // let start_saturation = color1.1.min(color2.1);
    // let end_saturation = color1.1.max(color2.1);
    let start_saturation = color1.1;
    let end_saturation = color2.1;
    let saturation_step = (end_saturation - start_saturation) / char_count as f64;

    // let start_value = color1.2.min(color2.2);
    // let end_value = color1.2.max(color2.2);
    let start_value = color1.2;
    let end_value = color2.2;
    let value_step = (end_value - start_value) / char_count as f64;

    for ch in text.chars() {
        if ch.is_whitespace() {
            result.push(ch);
        } else {
            let ansi = ansi_hsv(hue, saturation, value);
            
            result.push_str(&ansi);
            result.push(ch);
            result.push_str(&ansi_reset());

            hue += hue_step;
            saturation += saturation_step;
            value += value_step;
        }
    }

    result
}

fn rgb_text(text: impl Display, color: (u8, u8, u8)) -> String {
    let text = text.to_string();
    let (hue, saturation, value) = color;
    let ansi = ansi_rgb(hue, saturation, value);
    let ansi_reset = ansi_reset();

    format!("{}{}{}", ansi, text, ansi_reset)
}

fn clear_screen() {
    print!("\x1b[2J\x1b[1;1H");
}

#[allow(dead_code)]
fn gradient_lines(text: impl Display, color1: (f64, f64, f64), color2: (f64, f64, f64)) -> String {
    let mut result = String::new();
    let text = text.to_string();
    

    let (hue, saturation, value) = color1;
    let lines_count = text.lines().count();
    // let start_hue = color1.0.min(color2.0);
    // let end_hue = color1.0.max(color2.0);
    let start_hue = color1.0;
    let end_hue = color2.0;
    let hue_step = (end_hue - start_hue) / lines_count as f64;

    // let start_saturation = color1.1.min(color2.1);
    // let end_saturation = color1.1.max(color2.1);
    let start_saturation = color1.1;
    let end_saturation = color2.1;
    let saturation_step = (end_saturation - start_saturation) / lines_count as f64;

    // let start_value = color1.2.min(color2.2);
    // let end_value = color1.2.max(color2.2);
    let start_value = color1.2;
    let end_value = color2.2;
    let value_step = (end_value - start_value) / lines_count as f64;

    let mut line_hue = hue;
    let mut line_saturation = saturation;
    let mut line_value = value;

    for ch in text.chars() {
        if ch == '\n' {
            line_hue += hue_step;
            line_saturation += saturation_step;
            line_value += value_step;
            result.push(ch);
        } else {
            if ch.is_whitespace() {
                result.push(ch);
            } else {
                let ansi = ansi_hsv(line_hue, line_saturation, line_value);
                
                result.push_str(&ansi);
                result.push(ch);
                result.push_str(&ansi_reset());

            }
        }
    }

    result
}



fn main() {
    // Set up logging with the `tracing` crate, with debug level logging.
    let _ = tracing_subscriber::fmt::SubscriberBuilder::default()
        .with_max_level(tracing::Level::INFO)
        .without_time()
        .init();

    // If we're in debug mode, start the compilation in a separate thread.
    // This is to allow the process to have more stack space.
    let stack_size_mb = if cfg!(debug_assertions) {
        DEBUG_STACK_SIZE_MB
    } else {
        RELEASE_STACK_SIZE_MB
    };

    let child = std::thread::Builder::new()
        .stack_size(stack_size_mb * 1024 * 1024)
        .spawn(cli)
        .unwrap();

    // Wait for the thread to finish.
    child.join().unwrap();

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
    /*
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
"#, &mut env).unwrap();
*/

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