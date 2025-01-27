
use reckon::*;
use clap::Parser;
use std::fmt::{Display, Write};
use std::fs::read_to_string;
use tracing::{info, error};

// The stack sizes of the threads used to compile the code.
const RELEASE_STACK_SIZE_MB: usize = 512;
const DEBUG_STACK_SIZE_MB: usize = RELEASE_STACK_SIZE_MB;

const ABOUT: &str = "┌─────────────────────About──────────────────────┐\n           Hello, welcome to \u{1b}[93mReckon\u{1b}[m\u{1b}[0m!  \n       Written by: http://adam-mcdaniel.net       \n                                                  \n I wrote Reckon to be an easy to use, convenient  \n tool for writing programs like computer algebra  \n   systems, graph analyzers, or type-checkers!    \n└────────────────────────────────────────────────┘\n┌────────────────────Features────────────────────┐\n Reckon is a simple logic🧮 programming language   \n designed to be used either as a standalone tool  \n or as a library in a larger Rust project for     \n  performing logical inference💡 and reasoning.    \n                                                  \n It supports:                                     \n ☞ Rules, facts, and queries with horn clauses📯   \n ☞ Negation as failure🚫                           \n ☞ Unification and backtracking🔀                  \n ☞ Configurable search strategies🔎                \n ☞ A REPL for interactive use🔄                    \n ☞ Importing programs from files📂                 \n ☞ A simple interface as a library📚               \n└────────────────────────────────────────────────┘\n┌────────────────About the Author────────────────┐\n I'm a Comp💻 Sci🧪 PhD student at the University   \n of Tennessee. I'm in love with language design,  \n         compilers, and writing software!         \n                                                  \n      Check out my other projects on GitHub:      \n         https://github.com/adam-mcdaniel         \n└────────────────────────────────────────────────┘";

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
    println!("Welcome to...\n");

    print_logo(BG_2);

    println!("\nUse the REPL to enter your rules, queries, and search options.\nFor help, input \":help\"\n");
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
                        } else if line == ":a" || line == ":about" {
                            println!("{}", ABOUT);
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
                            println!("  :s, :save    - Push the current environment to be restored");
                            println!("  :r, :restore - Pop the last environment pushed");
                            println!("  :x, :examine - Print the environment");
                            println!("  :c, :clear   - Reset the environment");
                            println!("  :h, :help    - Show this help message");
                            continue;
                        }
                    }

                    if (line.trim().len() >= 2 && &line.trim()[0..2] == "//")
                        || line.trim().chars().next() == Some('%') {
                        continue;
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
fn ansi_rgb_bg(r: u8, g: u8, b: u8) -> String {
    format!("\x1b[48;2;{};{};{}m", r, g, b)
}

#[allow(dead_code)]
fn ansi_hsv_bg(h: f64, s: f64, v: f64) -> String {
    let (r, g, b) = hsv_to_rgb((h, s, v));
    ansi_rgb_bg(r, g, b)
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
fn gradient(text: impl Display, color1: (f64, f64, f64), color2: (f64, f64, f64), fg: bool) -> String {
    let mut result = String::new();
    let text = text.to_string();
    let text = strip_ansi_codes(&text);
    

    let (mut hue, mut saturation, mut value) = color1;
    let char_count = text.chars().filter(|c| !c.is_whitespace() || !fg).count();
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
        if ch == '\n' {
            // Print reset
            result.push_str(&ansi_reset());
            result.push(ch);
        } else if ch.is_whitespace() && fg {
            result.push(ch);
        } else {
            let ansi = if fg {
                ansi_hsv(hue, saturation, value)
            } else {
                ansi_hsv_bg(hue, saturation, value)
            };
            
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

fn ascii_art_get_size(art: &str) -> (usize, usize) {
    let plain_art = strip_ansi_codes(art);
    let lines: Vec<&str> = plain_art.lines().collect();
    let width = lines.iter().map(|line| line.chars().count()).max().unwrap_or(0);

    (width, lines.len())
}

fn pad_ascii_art(art: &str, new_width: usize, new_height: usize) -> String {
    let mut result = String::new();
    let plain_art = strip_ansi_codes(art);
    let lines: Vec<&str> = art.lines().collect();
    let plain_lines: Vec<&str> = plain_art.lines().collect();

    let (_, height) = ascii_art_get_size(art);

    let height_diff = new_height - height;

    for _ in 0..height_diff / 2 {
        for _ in 0..new_width {
            result.push(' ');
        }
        result.push('\n');
    }

    // for line in lines {
    for (line, plain_line) in lines.iter().zip(plain_lines.iter()) {
        let plain_line_width = plain_line.chars().count();
        let mut remaining_line_width = new_width - plain_line_width;
        for _ in 0..remaining_line_width / 2 {
            remaining_line_width -= 1;
            result.push(' ');
        }
        result.push_str(line);
        while remaining_line_width > 0 {
            remaining_line_width -= 1;
            result.push(' ');
        }

        result.push('\n');
    }

    while result.lines().count() < new_height {
        for _ in 0..new_width {
            result.push(' ');
        }
        result.push('\n');
    }

    let (test_width, test_height) = ascii_art_get_size(&result);
    if test_width != new_width || test_height != new_height {
        panic!("Padding failed: expected ({}, {}), got ({}, {})", new_width, new_height, test_width, test_height);
    }

    result
}

fn crop_ascii_art(art: &str, new_width: usize, new_height: usize) -> String {
    let mut result = String::new();
    let plain_art = strip_ansi_codes(art);
    let plain_lines: Vec<&str> = plain_art.lines().collect();

    let (width, height) = ascii_art_get_size(&plain_art);

    let width_diff = width - new_width;
    let height_diff = height - new_height;

    for plain_line in plain_lines.iter().skip(height_diff/2).take(new_height) {
        for ch in plain_line.chars().skip(width_diff/2).take(new_width) {
            result.push(ch);
        }

        result.push('\n');
    }

    let (test_width, test_height) = ascii_art_get_size(&result);
    if test_width != new_width || test_height != new_height {
        panic!("Cropping failed: expected ({}, {}), got ({}, {})", new_width, new_height, test_width, test_height);
    }

    result
}

#[derive(Debug, Clone, Copy)]
struct Border {
    top_left: char,
    top_right: char,
    bottom_left: char,
    bottom_right: char,
    horizontal: char,
    vertical: char,
}

fn strip_ansi_codes(text: &str) -> String {
    let mut result = String::new();
    let mut chars = text.chars();
    while let Some(ch) = chars.next() {
        if ch == '\x1b' {
            while let Some(ch) = chars.next() {
                if ch == 'm' {
                    break;
                }
            }
        } else {
            result.push(ch);
        }
    }

    result
}

fn ascii_art_with_border(art: &str, border: Border) -> String {
    let mut result = String::new();
    let lines: Vec<&str> = art.lines().collect();
    let plain_art = strip_ansi_codes(art);
    let plain_lines: Vec<&str> = plain_art.lines().collect();
    
    let width = plain_art.lines().map(|line| line.chars().count()).max().unwrap_or(0);

    result.push_str(&ansi_reset());
    result.push(border.top_left);
    for _ in 0..width {
        result.push_str(&ansi_reset());
        result.push(border.horizontal);
    }
    result.push_str(&ansi_reset());
    result.push(border.top_right);
    result.push('\n');

    for (line, plain_lines) in lines.iter().zip(plain_lines.iter()) {
        result.push_str(&ansi_reset());
        result.push(border.vertical);
        result.push_str(line);
        for _ in 0..width - plain_lines.chars().count() {
            result.push(' ');
        }
        result.push_str(&ansi_reset());
        result.push(border.vertical);
        result.push('\n');
    }

    result.push_str(&ansi_reset());
    result.push(border.bottom_left);
    for _ in 0..width {
        result.push_str(&ansi_reset());
        result.push(border.horizontal);
    }
    result.push_str(&ansi_reset());
    result.push(border.bottom_right);
    result.push('\n');

    result
}

fn get_last_ansi_code_at(art: &str, row: usize, col: usize) -> Option<String> {
    let mut result = String::new();
    let mut chars = art.chars();
    let mut current_row = 0;
    let mut current_col = 0;
    let mut since_last_ansi = 0;
    while let Some(ch) = chars.next() {
        if current_row >= row && current_col >= col {
            break;
        }

        if ch == '\x1b' {
            result.clear();
            result.push(ch);
            while let Some(ch) = chars.next() {
                result.push(ch);
                if ch == 'm' {
                    break;
                }
            }
            since_last_ansi = 0;
        } else {
            if ch == '\n' {
                current_row += 1;
                current_col = 0;
                result.clear();
            } else {
                current_col += 1;
                since_last_ansi += 1;
            }
        }
    }

    if result.is_empty() || since_last_ansi > 1 {
        None
    } else {
        Some(result)
    }
}

fn reapply_ansi_to_plain(template: &str, colored: &str) -> String {
    let mut result = String::new();
    
    let plain_template = strip_ansi_codes(template);

    let (mut row, mut col) = (0, 0);
    for ch in plain_template.chars() {
        if ch == '\n' {
            row += 1;
            col = 0;
        } else {
            col += 1;
        }

        if let Some(ansi) = get_last_ansi_code_at(colored, row, col) {
            result.push_str(&ansi);
        } else if let Some(ansi) = get_last_ansi_code_at(template, row, col) {
            result.push_str(&ansi);
        } else {
            result.push_str("\x1b[0m");
        }

        result.push(ch);
    }

    /*
    while colored_chars.peek().is_some() {
        let colored_char = colored_chars.next().unwrap();
        if let Some(plain_char) = plain_chars.peek() {
            if *plain_char == '\x1b' {

                // Get the ANSI code
                let mut ansi_code = String::new();
                ansi_code.push(*plain_char);
                while let Some(ch) = plain_chars.next() {
                    ansi_code.push(ch);
                    if ch == 'm' {
                        break;
                    }
                }
                result.push_str(&ansi_code);
            }
        }

        if colored_char == '\x1b' {
            // Get the ANSI code
            let mut ansi_code = String::new();
            ansi_code.push(colored_char);
            while let Some(ch) = colored_chars.next() {
                ansi_code.push(ch);
                if ch == 'm' {
                    break;
                }
            }
            result.push_str(&ansi_code);
        } else if let Some(plain_char) = plain_chars.next() {
            result.push(plain_char);
        }
    }
    */


    result
}

fn ascii_art_fg_bg(fg_art: &str, bg_art: &str) -> String {
    let mut result = String::new();
    let (bg_width, bg_height) = ascii_art_get_size(bg_art);

    let bg_art = pad_ascii_art(bg_art, bg_width, bg_height);
    let fg_art = pad_ascii_art(fg_art, bg_width, bg_height);
    
    let plain_fg_art = strip_ansi_codes(&fg_art);
    let plain_bg_art = strip_ansi_codes(&bg_art);

    if plain_fg_art.chars().count() != plain_bg_art.chars().count() {
        panic!("Art lengths do not match: {} vs {}", plain_fg_art.chars().count(), plain_bg_art.chars().count());
    }

    for (fg_char, bg_char) in plain_fg_art.chars().zip(plain_bg_art.chars()) {
        if fg_char.is_whitespace() {
            result.push(bg_char);
        } else {
            result.push(fg_char);
        }
    }

    // Reapply the background ANSI codes
    result = reapply_ansi_to_plain(&result, &bg_art);
    result = reapply_ansi_to_plain(&result, &fg_art);

    result
}

fn print_logo(bg: &str) {
    let border = Border {
        top_left: '╔',
        top_right: '╗',
        bottom_left: '╚',
        bottom_right: '╝',
        horizontal: '═',
        vertical: '║',
    };
    let mut logo = COLORED_LOGO_1.to_string();
    let (width, height) = ascii_art_get_size(&logo);
    
    logo = pad_ascii_art(&logo, width + 2, height + 2);
    logo = ascii_art_with_border(&logo, border);
    logo = logo.replace(' ', "\x08");
    logo = format!("{}{}", ansi_reset(), logo);
    if !bg.is_empty() {
        let (fg_width, fg_height) = ascii_art_get_size(&logo);
        
        let bg_width = fg_width + 6;
        let bg_height = fg_height + 6;
    
        let bg = crop_ascii_art(bg, bg_width, bg_height);
    
        // Apply colors to the bg
        // let bg = rgb_text(&bg, (255, 255, 255));
        let bg = gradient(&bg, (0.0, 1.0, 1.0), (240.0, 1.0, 1.0), true);
    
    
        logo = ascii_art_fg_bg(&logo, &bg);
    }

    logo = logo.replace('\x08', " ");
    println!("{}", logo);

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
}

#[allow(dead_code)]
const BG_1: &str = r#" _______________________________________________________________________ 
|       (_      (_      (_      (_      (_      (_      (_      (_      |
|        _)      _)      _)      _)      _)      _)      _)      _)     |
|  _   _(  _   _(  _   _(  _   _(  _   _(  _   _(  _   _(  _   _(  _   _|
|_| |_| |_| |_| |_| |_| |_| |_| |_| |_| |_| |_| |_| |_| |_| |_| |_| |_| |
|      _)      _)      _)      _)      _)      _)      _)      _)       |
|     (_      (_      (_      (_      (_      (_      (_      (_        |
|_   _  )_   _  )_   _  )_   _  )_   _  )_   _  )_   _  )_   _  )_   _  |
| |_| |_| |_| |_| |_| |_| |_| |_| |_| |_| |_| |_| |_| |_| |_| |_| |_| |_|
|       (_      (_      (_      (_      (_      (_      (_      (_      |
|        _)      _)      _)      _)      _)      _)      _)      _)     |
|  _   _(  _   _(  _   _(  _   _(  _   _(  _   _(  _   _(  _   _(  _   _|
|_| |_| |_| |_| |_| |_| |_| |_| |_| |_| |_| |_| |_| |_| |_| |_| |_| |_| |
|      _)      _)      _)      _)      _)      _)      _)      _)       |
|     (_      (_      (_      (_      (_      (_      (_      (_        |
|_   _  )_   _  )_   _  )_   _  )_   _  )_   _  )_   _  )_   _  )_   _  |
| |_| |_| |_| |_| |_| |_| |_| |_| |_| |_| |_| |_| |_| |_| |_| |_| |_| |_|
|       (_      (_      (_      (_      (_      (_      (_      (_      |
|        _)      _)      _)      _)      _)      _)      _)      _)     |
|_______(_______(_______(_______(_______(_______(_______(_______(_______|"#;

#[allow(dead_code)]
const BG_2: &str = r#" / __ \ \__/ / __ \ \__/ / __ \ \__/ / __ \ \__/ / __ \ \__/ / __ \ \_
/ /  \ \____/ /  \ \____/ /  \ \____/ /  \ \____/ /  \ \____/ /  \ \__
\ \__/ / __ \ \__/ / __ \ \__/ / __ \ \__/ / __ \ \__/ / __ \ \__/ / _
 \____/ /  \ \____/ /  \ \____/ /  \ \____/ /  \ \____/ /  \ \____/ / 
 / __ \ \__/ / __ \ \__/ / __ \ \__/ / __ \ \__/ / __ \ \__/ / __ \ \_
/ /  \ \____/ /  \ \____/ /  \ \____/ /  \ \____/ /  \ \____/ /  \ \__
\ \__/ / __ \ \__/ / __ \ \__/ / __ \ \__/ / __ \ \__/ / __ \ \__/ / _
 \____/ /  \ \____/ /  \ \____/ /  \ \____/ /  \ \____/ /  \ \____/ / 
 / __ \ \__/ / __ \ \__/ / __ \ \__/ / __ \ \__/ / __ \ \__/ / __ \ \_
/ /  \ \____/ /  \ \____/ /  \ \____/ /  \ \____/ /  \ \____/ /  \ \__
\ \__/ / __ \ \__/ / __ \ \__/ / __ \ \__/ / __ \ \__/ / __ \ \__/ / _
 \____/ /  \ \____/ /  \ \____/ /  \ \____/ /  \ \____/ /  \ \____/ / 
 / __ \ \__/ / __ \ \__/ / __ \ \__/ / __ \ \__/ / __ \ \__/ / __ \ \_
/ /  \ \____/ /  \ \____/ /  \ \____/ /  \ \____/ /  \ \____/ /  \ \__
\ \__/ / __ \ \__/ / __ \ \__/ / __ \ \__/ / __ \ \__/ / __ \ \__/ / _
 \____/ /  \ \____/ /  \ \____/ /  \ \____/ /  \ \____/ /  \ \____/ / 
 / __ \ \__/ / __ \ \__/ / __ \ \__/ / __ \ \__/ / __ \ \__/ / __ \ \_
/ /  \ \____/ /  \ \____/ /  \ \____/ /  \ \____/ /  \ \____/ /  \ \__
\ \__/ / __ \ \__/ / __ \ \__/ / __ \ \__/ / __ \ \__/ / __ \ \__/ / _
 \____/ /  \ \____/ /  \ \____/ /  \ \____/ /  \ \____/ /  \ \____/ / 
 / __ \ \__/ / __ \ \__/ / __ \ \__/ / __ \ \__/ / __ \ \__/ / __ \ \_
/ /  \ \____/ /  \ \____/ /  \ \____/ /  \ \____/ /  \ \____/ /  \ \__"#;
    // println!("{}", ascii_art_with_border(COLORED_LOGO_1, border));
    // println!("{}", ascii_art_with_border(COLORED_LOGO_1, border2));
    // println!("{}", ascii_art_with_border(&pad_ascii_art(COLORED_LOGO_1, w + 4, h + 4), border2));
    
#[allow(dead_code)]
const BG_3: &str = r#"\_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_
/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ 
\_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_
/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ 
\_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_
/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ 
\_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_
/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ 
\_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_
/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ 
\_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_
/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ 
\_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_
/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ 
\_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_
/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ 
\_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_
/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/"#;

#[allow(dead_code)]
const BG_4: &str = r#"  \__   \__   \__   \__   \__   \__   \__   \__   \__   \__   \__   \_
__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/ 
  \     \     \     \     \     \     \     \     \     \     \     \
__/   __/   __/   __/   __/   __/   __/   __/   __/   __/   __/   __/
  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \_
__/     /     /     /     /     /     /     /     /     /     /     / 
  \__   \__   \__   \__   \__   \__   \__   \__   \__   \__   \__   \_
__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/ 
  \     \     \     \     \     \     \     \     \     \     \     \
__/   __/   __/   __/   __/   __/   __/   __/   __/   __/   __/   __/
  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \_
__/     /     /     /     /     /     /     /     /     /     /     / 
  \__   \__   \__   \__   \__   \__   \__   \__   \__   \__   \__   \_
__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/ 
  \     \     \     \     \     \     \     \     \     \     \     \
__/   __/   __/   __/   __/   __/   __/   __/   __/   __/   __/   __/
  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \_
__/     /     /     /     /     /     /     /     /     /     /     / 
  \__   \__   \__   \__   \__   \__   \__   \__   \__   \__   \__   \_
__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/ 
  \     \     \     \     \     \     \     \     \     \     \     \
__/   __/   __/   __/   __/   __/   __/   __/   __/   __/   __/   __/
  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \__/  \_"#;