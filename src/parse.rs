use super::*;

use nom::{
    branch::alt,
    bytes::complete::{is_not, tag, take_while1, take_while_m_n},
    character::complete::{char, multispace0, multispace1, one_of},
    combinator::{cut, map, map_opt, map_res, opt, value, verify},
    error::{FromExternalError, ParseError},
    multi::{fold_many0, separated_list0, separated_list1},
    sequence::{delimited, preceded, tuple},
    IResult, Parser,
};

use crate::{
    AppTerm,
    Query,
    Rule,
    Term,
    Var,
};

type Error<'a> = nom::error::VerboseError<&'a str>;

// A small utility to strip leading/trailing whitespace around a parser
fn ws<'a, F: 'a, O>(inner: F) -> impl FnMut(&'a str) -> IResult<&'a str, O, Error<'a>>
where
    F: FnMut(&'a str) -> IResult<&'a str, O, Error<'a>>,
{
    delimited(multispace0, inner, multispace0)
}

// --- Base parsers for atoms ---

/// Parse an integer (i64). e.g. "-42", "123"
fn parse_int(i: &str) -> IResult<&str, i64, Error> {
    // 1. Optional sign
    // 2. Some digits
    let (i, num_str) = ws(map_res(recognize_sign_digits, |s: &str| s.parse::<i64>()))(i)?;
    Ok((i, num_str))
}

fn parse_int_term(i: &str) -> IResult<&str, Term, Error> {
    let (i, num) = parse_int(i)?;
    Ok((i, Term::Int(num)))
}

/// Recognize optional sign plus digits (used by parse_int).
fn recognize_sign_digits(i: &str) -> IResult<&str, &str, Error> {
    use nom::combinator::recognize;
    use nom::sequence::pair;

    recognize(pair(
        opt(one_of("+-")),
        take_while1(|c: char| c.is_ascii_digit()),
    ))(i)
}

/// Parse a boolean: `true` or `false`.
fn parse_bool(i: &str) -> IResult<&str, bool, Error> {
    alt((
        value(true, ws(tag("true"))),
        value(false, ws(tag("false"))),
    ))(i)
}

fn parse_bool_term(i: &str) -> IResult<&str, Term, Error> {
    let (i, b) = parse_bool(i)?;
    Ok((i, if b {
        Term::True
    } else {
        Term::False
    }))
}


/// Parse `nil`.
fn parse_nil(i: &str) -> IResult<&str, Term, Error> {
    value(Term::Nil, ws(tag("nil")))(i)
}

/// Parse the cut symbol `#`.
fn parse_cut(i: &str) -> IResult<&str, Term, Error> {
    value(Term::Cut, ws(tag("!")))(i)
}

// parser combinators are constructed from the bottom up:
// first we write parsers for the smallest elements (escaped characters),
// then combine them into larger parsers.

/// Parse a unicode sequence, of the form u{XXXX}, where XXXX is 1 to 6
/// hexadecimal numerals. We will combine this later with parse_escaped_char
/// to parse sequences like \u{00AC}.
fn parse_unicode<'a, E>(input: &'a str) -> IResult<&'a str, char, E>
where
    E: ParseError<&'a str> + FromExternalError<&'a str, std::num::ParseIntError>,
{
    // `take_while_m_n` parses between `m` and `n` bytes (inclusive) that match
    // a predicate. `parse_hex` here parses between 1 and 6 hexadecimal numerals.
    let parse_hex = take_while_m_n(1, 6, |c: char| c.is_ascii_hexdigit());

    // `preceded` takes a prefix parser, and if it succeeds, returns the result
    // of the body parser. In this case, it parses u{XXXX}.
    let parse_delimited_hex = preceded(
        char('u'),
        // `delimited` is like `preceded`, but it parses both a prefix and a suffix.
        // It returns the result of the middle parser. In this case, it parses
        // {XXXX}, where XXXX is 1 to 6 hex numerals, and returns XXXX
        delimited(char('{'), parse_hex, char('}')),
    );

    // `map_res` takes the result of a parser and applies a function that returns
    // a Result. In this case we take the hex bytes from parse_hex and attempt to
    // convert them to a u32.
    let parse_u32 = map_res(parse_delimited_hex, move |hex| u32::from_str_radix(hex, 16));

    // map_opt is like map_res, but it takes an Option instead of a Result. If
    // the function returns None, map_opt returns an error. In this case, because
    // not all u32 values are valid unicode code points, we have to fallibly
    // convert to char with from_u32.
    map_opt(parse_u32, std::char::from_u32).parse(input)
}

/// Parse an escaped character: \n, \t, \r, \u{00AC}, etc.
fn parse_escaped_char<'a, E>(input: &'a str) -> IResult<&'a str, char, E>
where
    E: ParseError<&'a str> + FromExternalError<&'a str, std::num::ParseIntError>,
{
    preceded(
        char('\\'),
        // `alt` tries each parser in sequence, returning the result of
        // the first successful match
        alt((
            parse_unicode,
            // The `value` parser returns a fixed value (the first argument) if its
            // parser (the second argument) succeeds. In these cases, it looks for
            // the marker characters (n, r, t, etc) and returns the matching
            // character (\n, \r, \t, etc).
            value('\n', char('n')),
            value('\r', char('r')),
            value('\t', char('t')),
            value('\u{08}', char('b')),
            value('\u{0C}', char('f')),
            value('\\', char('\\')),
            value('/', char('/')),
            value('"', char('"')),
        )),
    )
    .parse(input)
}

/// Parse a backslash, followed by any amount of whitespace. This is used later
/// to discard any escaped whitespace.
fn parse_escaped_whitespace<'a, E: ParseError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, &'a str, E> {
    preceded(char('\\'), multispace1).parse(input)
}

/// Parse a non-empty block of text that doesn't include \ or "
fn parse_literal<'a, E: ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, &'a str, E> {
    // `is_not` parses a string of 0 or more characters that aren't one of the
    // given characters.
    let not_quote_slash = is_not("\"\\");

    // `verify` runs a parser, then runs a verification function on the output of
    // the parser. The verification function accepts out output only if it
    // returns true. In this case, we want to ensure that the output of is_not
    // is non-empty.
    verify(not_quote_slash, |s: &str| !s.is_empty()).parse(input)
}

/// A string fragment contains a fragment of a string being parsed: either
/// a non-empty Literal (a series of non-escaped characters), a single
/// parsed escaped character, or a block of escaped whitespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StringFragment<'a> {
    Literal(&'a str),
    EscapedChar(char),
    EscapedWS,
}

/// Combine parse_literal, parse_escaped_whitespace, and parse_escaped_char
/// into a StringFragment.
fn parse_fragment<'a, E>(input: &'a str) -> IResult<&'a str, StringFragment<'a>, E>
where
    E: ParseError<&'a str> + FromExternalError<&'a str, std::num::ParseIntError>,
{
    alt((
        // The `map` combinator runs a parser, then applies a function to the output
        // of that parser.
        map(parse_literal, StringFragment::Literal),
        map(parse_escaped_char, StringFragment::EscapedChar),
        value(StringFragment::EscapedWS, parse_escaped_whitespace),
    ))
    .parse(input)
}

/// Parse a string. Use a loop of parse_fragment and push all of the fragments
/// into an output string.
fn parse_string<'a, E>(input: &'a str) -> IResult<&'a str, String, E>
where
    E: ParseError<&'a str> + FromExternalError<&'a str, std::num::ParseIntError>,
{
    // fold is the equivalent of iterator::fold. It runs a parser in a loop,
    // and for each output value, calls a folding function on each output value.
    let build_string = fold_many0(
        // Our parser function – parses a single string fragment
        parse_fragment,
        // Our init value, an empty string
        String::new,
        // Our folding function. For each fragment, append the fragment to the
        // string.
        |mut string, fragment| {
            match fragment {
                StringFragment::Literal(s) => string.push_str(s),
                StringFragment::EscapedChar(c) => string.push(c),
                StringFragment::EscapedWS => {}
            }
            string
        },
    );

    // Finally, parse the string. Note that, if `build_string` could accept a raw
    // " character, the closing delimiter " would never match. When using
    // `delimited` with a looping parser (like fold), be sure that the
    // loop won't accidentally match your closing delimiter!
    delimited(char('"'), build_string, char('"')).parse(input)
}

fn parse_string_term(i: &str) -> IResult<&str, Term, Error> {
    let (i, s) = parse_string(i)?;
    Ok((i, Term::Str(s)))
}

/// Parse a variable token: starts with uppercase or underscore, followed by alphanums or `_`.
/// Example: `X`, `User42`, `_foo`.
fn parse_var(i: &str) -> IResult<&str, Term, Error> {
    // Allowed first char: [A-Z_]
    // Allowed subsequent: [A-Za-z0-9_]
    let (i, var_name) = ws(map_res(
        take_while1(|c: char| c.is_ascii_alphanumeric() || c == '_'),
        |s: &str| {
            // We treat it as a Var if the first char is uppercase or '_'
            let first_char = s.chars().next().unwrap();
            if first_char.is_uppercase() || first_char == '_' {
                Ok(s.to_string())
            } else {
                Err("Not a variable")
            }
        },
    ))(i)?;
    // Turn it into a Term::Var
    Ok((i, Term::Var(Var::new(var_name))))
}

/// Parse a lowercase identifier (symbol).
/// Example: `foo`, `bar_baz42`.
fn parse_symbol(i: &str) -> IResult<&str, String, Error> {
    // Allowed first char: [a-z]
    // Allowed subsequent: [A-Za-z0-9_]
    let (i, sym) = ws(
        take_while1(|c: char| c.is_ascii_alphanumeric() || c == '_'),
    )(i)?;
    Ok((i, sym.to_string()))
}

// --- Structured term parsers ---

/// Parse a cons cell: `[term | term]`
/// This is a special case of a list, where the tail is a single term.
fn parse_cons(i: &str) -> IResult<&str, Term, Error> {
    let (i, _) = ws(char('['))(i)?;
    let (i, (head, tail)) = match opt(tuple((ws(parse_term), ws(char('|')), ws(parse_term))))(i)? {
        (i, Some((head, _, tail))) => (i, (head, tail)),
        (i, None) => {
            // Try to parse a single element list
            match ws(parse_term)(i) {
                Ok((i, head)) => (i, (head, Term::Nil)),
                Err(_) => (i, (Term::Nil, Term::Nil)),
            }
        }
    };
    let (i, _) = ws(char(']'))(i)?;
    Ok((i, Term::Cons(Box::new(head), Box::new(tail))))
}

/// Parse an application term: `foo(term, term, ...)`
/// We parse the symbol first, then arguments in parentheses.
fn parse_app(i: &str) -> IResult<&str, Term, Error> {
    // parse the function name
    let (i, func_str) = parse_symbol(i)?;
    let (i, args) = delimited(
        ws(char('(')),
        cut(separated_list0(ws(char(',')), cut(parse_term))),
        ws(char(')')),
    )(i)?;

    let app_term = AppTerm::new(func_str, args);
    Ok((i, Term::App(app_term)))
}

// --- Catch-all term parser ---

/// The master parser for any `Term`.
fn parse_term_atom(i: &str) -> IResult<&str, Term, Error> {
    alt((
        parse_app,
        parse_cut,
        parse_nil,
        parse_bool_term,
        parse_int_term,
        parse_string_term,
        parse_var,
        parse_cons,
        map(parse_symbol, |x| Term::Sym(Symbol::from(x))),
        // parse_set,
        // parse_map,
    ))(i)
}

fn parse_complement(i: &str) -> IResult<&str, Term, Error> {
    let (i, _) = ws(tag("~"))(i)?;
    let (i, term) = parse_term_atom(i)?;
    Ok((i, term.negate()))
}

pub(super) fn parse_term(i: &str) -> IResult<&str, Term, Error> {
    alt((parse_complement, parse_term_atom))(i)
}

// --- Rule parsing ---

/// Parse a single `Rule`: `head :- goal1, goal2, ... .`
/// or a fact: `head.`
fn parse_app_rule(i: &str) -> IResult<&str, Rule, Error> {
    // We'll parse an app term as the head, check for optional ":-" tail,
    // then end with a period.
    let (i, head_term) = ws(parse_app)(i)?;

    // The head must be an AppTerm (Term::App). If not, it's an error.
    let head_app = match head_term {
        Term::App(a) => a,
        _ => {
            return Err(nom::Err::Error(nom::error::VerboseError {
                errors: vec![(
                    i,
                    nom::error::VerboseErrorKind::Nom(nom::error::ErrorKind::Tag),
                )],
            }))
        }
    };

    // Possibly parse ":-" tail or skip.
    let (i, tail_terms) = opt(preceded(
        ws(tag(":-")),
        separated_list1(ws(char(',')), parse_term),
    ))(i)?;

    // Expect a dot at the end
    let (i, _) = ws(char('.'))(i)?;

    let tail = tail_terms.unwrap_or_else(Vec::new);
    Ok((
        i,
        Rule {
            head: head_app.into(),
            tail,
        },
    ))
}

fn parse_term_rule(i: &str) -> IResult<&str, Rule, Error> {
    // We'll parse a term as the head, check for optional ":-" tail,
    // then end with a period.
    let (i, head_term) = ws(parse_term)(i)?;

    // Possibly parse ":-" tail or skip.
    let (i, tail_terms) = opt(preceded(
        ws(tag(":-")),
        separated_list1(ws(char(',')), parse_term),
    ))(i)?;

    // Expect a dot at the end
    let (i, _) = ws(char('.'))(i)?;

    let tail = tail_terms.unwrap_or_else(Vec::new);
    Ok((
        i,
        Rule {
            head: head_term.into(),
            tail,
        },
    ))
}

/// Parse a single `Rule`: `head :- goal1, goal2, ... .`
/// or a fact: `head.`
pub(super) fn parse_rule(i: &str) -> IResult<&str, Rule, Error> {
    alt((parse_app_rule, cut(parse_term_rule)))(i)
}

// --- Query parsing ---
//
// A query might be: `?- goal1, goal2, ... .`
// We'll parse a question mark, dash, then a list of terms separated by commas, ended by a period.

pub(super) fn parse_query(i: &str) -> IResult<&str, Query, Error> {
    let (i, _) = ws(tag("?-"))(i)?;
    let (i, goals) = separated_list1(ws(char(',')), parse_term)(i)?;
    let (i, _) = ws(char('.'))(i)?;
    Ok((i, Query::new(goals)))
}

fn options_int_flag(i: &str) -> IResult<&str, (&str, String, i64), Error> {
    let flag_input = i;
    let (i, flag_name) = ws(parse_symbol)(i)?;
    let (i, _) = ws(char('='))(i)?;
    let (i, value) = ws(parse_int)(i)?;

    Ok((i, (flag_input, flag_name, value)))
}

fn options_bool_flag(i: &str) -> IResult<&str, (&str, String, bool), Error> {
    let flag_input = i;
    let (i, flag_name) = ws(parse_symbol)(i)?;
    let (i, _) = ws(char('='))(i)?;
    let (i, value) = ws(parse_bool)(i)?;

    Ok((i, (flag_input, flag_name, value)))
}

fn options_str_flag(i: &str) -> IResult<&str, (&str, String, String), Error> {
    let flag_input = i;
    let (i, flag_name) = ws(parse_symbol)(i)?;
    let (i, _) = ws(char('='))(i)?;
    let (i, value) = ws(parse_string)(i)?;

    Ok((i, (flag_input, flag_name, value)))
}

pub(super) fn parse_search_config<'a, 'b, S>(i: &'a str, existing_config: &'b mut SearchConfig<S>) -> IResult<&'a str, (), Error<'a>> where S: Solver {
    // Parse the search config
    let (i, _) = ws(tag("options"))(i)?;
    let (i, _) = ws(tag("("))(i)?;
    // let options_int_flag = |flag_name| {
    //     let (i, _) = ws(tag(flag_name))(i)?;
    //     let (i, _) = ws(char('='))(i)?;
    //     let (i, value) = ws(parse_int)(i)?;
    //     Ok((i, value))
    // };

    // let options_bool_flag = |flag_name| {
    //     let (i, _) = ws(tag(flag_name))(i)?;
    //     let (i, _) = ws(char('='))(i)?;
    //     let (i, value) = ws(boolean)(i)?;
    //     Ok((i, value))
    // };

    // let options_str_flag = |flag_name| {
    //     let (i, _) = ws(tag(flag_name))(i)?;
    //     let (i, _) = ws(char('='))(i)?;
    //     let (i, value) = ws(parse_string)(i)?;
    //     Ok((i, value))
    // };

    enum ConfigValue {
        Int(i64),
        Bool(bool),
        Str(String),
    }

    let (i, flags) = cut(separated_list0(ws(char(',')), alt((
        map(options_int_flag, |(flag_input, name, value)| (flag_input, name, ConfigValue::Int(value))),
        map(options_bool_flag, |(flag_input, name, value)| (flag_input, name, ConfigValue::Bool(value))),
        map(options_str_flag, |(flag_input, name, value)| (flag_input, name, ConfigValue::Str(value))),
    ))))(i)?;

    let (i, _) = cut(ws(char(')')))(i)?;
    let (i, _) = cut(ws(char('.')))(i)?;

    let mut config = existing_config.clone();
    for (i, flag, value) in flags {
        match value {
            ConfigValue::Int(value) => {
                config = match flag.as_str() {
                    "step_limit" => config.with_step_limit(value as usize),
                    "depth_limit" => config.with_depth_limit(value as usize),
                    "width_limit" => config.with_width_limit(value as usize),
                    "solution_limit" => config.with_solution_limit(value as usize),
                    _ => {
                        return Err(nom::Err::Error(nom::error::VerboseError {
                            errors: vec![(
                                i,
                                nom::error::VerboseErrorKind::Nom(nom::error::ErrorKind::Tag),
                            )],
                        }))
                    }
                }
            },
            ConfigValue::Bool(value) => {
                config = match flag.as_str() {
                    "pruning" => config.with_pruning(value),
                    "require_rule_head_match" => config.with_require_rule_head_match(value),
                    "reduce_query" => config.with_reduce_query(value),
                    "clean_memoization" => config.with_clean_memoization(value),
                    "stop_after_first_goal" => config.with_stop_after_first_goal(value),
                    "enable_sorting" => config.with_sorting_enabled(value),
                    _ => {
                        return Err(nom::Err::Error(nom::error::VerboseError {
                            errors: vec![(
                                i,
                                nom::error::VerboseErrorKind::Nom(nom::error::ErrorKind::Tag),
                            )],
                        }))
                    }
                }
            },
            ConfigValue::Str(value) => {
                config = match flag.as_str() {
                    "traversal" => {
                        let traversal = match value.as_str() {
                            "breadth_first" => Traversal::BreadthFirst,
                            "depth_first" => Traversal::DepthFirst,
                            _ => {
                                return Err(nom::Err::Error(nom::error::VerboseError {
                                    errors: vec![(
                                        i,
                                        nom::error::VerboseErrorKind::Nom(nom::error::ErrorKind::Tag),
                                    )],
                                }))
                            }
                        };
                        config.with_traversal(traversal)
                    },
                    _ => {
                        return Err(nom::Err::Error(nom::error::VerboseError {
                            errors: vec![(
                                i,
                                nom::error::VerboseErrorKind::Nom(nom::error::ErrorKind::Tag),
                            )],
                        }))
                    }
                }
            }
        }
    }

    *existing_config = config;

    Ok((i, ()))
}

#[cfg(test)]
mod test {
    use super::*;
/*
    #[test]
    fn test_parse_int() {
        // Test positive integer
        let input = "42";
        let expected = Term::Int(42);
        let (_, result) = parse_term(input).unwrap();
        assert_eq!(result, expected);

        // Test negative integer
        let input = "-10";
        let expected = Term::Int(-10);
        let (_, result) = parse_term(input).unwrap();
        assert_eq!(result, expected);

        // Test zero
        let input = "0";
        let expected = Term::Int(0);
        let (_, result) = parse_term(input).unwrap();
        assert_eq!(result, expected);

        // Test application with integer
        let input = "foo(2)";
        let expected = Term::App(AppTerm::new("foo", vec![Term::Int(2)]));
        let (_, result) = parse_term(input).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_parse_bool() {
        let input = "true";
        let expected = Term::True;
        let (_, result) = parse_term(input).unwrap();
        assert_eq!(result, expected);

        let input = "false";
        let expected = Term::False;
        let (_, result) = parse_term(input).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_parse_string() {
        let input = "\"hello world\"";
        let expected = Term::Str("hello world".to_string());
        let (_, result) = parse_term(input).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_parse_var() {
        let input = "X";
        let expected = Term::Var(Var::new("X"));
        let (_, result) = parse_term(input).unwrap();
        assert_eq!(result, expected);

        let input = "_foo";
        let expected = Term::Var(Var::new("_foo"));
        let (_, result) = parse_term(input).unwrap();
        assert_eq!(result, expected);

        let input = "User42";
        let expected = Term::Var(Var::new("User42"));
        let (_, result) = parse_term(input).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_parse_nil() {
        let input = "nil";
        let expected = Term::Nil;
        let (_, result) = parse_term(input).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_parse_cut() {
        let input = "!";
        let expected = Term::Cut;
        let (_, result) = parse_term(input).unwrap();
        assert_eq!(result, expected);
    }

    // #[test]
    // fn test_parse_set() {
    //     let input = "{1, 2, 3}";
    //     let mut set = HashSet::new();
    //     set.insert(Term::Int(1));
    //     set.insert(Term::Int(2));
    //     set.insert(Term::Int(3));
    //     let expected = Term::Set(set);
    //     let (_, result) = parse_term(input).unwrap();
    //     assert_eq!(result, expected);

    //     let input = "{X, true, \"hello\"}";
    //     let mut set = HashSet::new();
    //     set.insert(Term::Var(Var::new("X")));
    //     set.insert(Term::True);
    //     set.insert(Term::Str("hello".to_string()));
    //     let expected = Term::Set(set);
    //     let (_, result) = parse_term(input).unwrap();
    //     assert_eq!(result, expected);
    // }

    // #[test]
    // fn test_parse_map() {
    //     let input = "{\"foo\" => 42, \"bar\" => true}";
    //     let mut map = BTreeMap::new();
    //     map.insert(Term::Str("foo".to_string()), Term::Int(42));
    //     map.insert(Term::Str("bar".to_string()), Term::True);
    //     let expected = Term::Map(map);
    //     let (_, result) = parse_term(input).unwrap();
    //     assert_eq!(result, expected);

    //     let input = "{X => Y, \"key\" => \"value\"}";
    //     let mut map = BTreeMap::new();
    //     map.insert(Term::Var(Var::new("X")), Term::Var(Var::new("Y")));
    //     map.insert(Term::Str("key".to_string()), Term::Str("value".to_string()));
    //     let expected = Term::Map(map);
    //     let (_, result) = parse_term(input).unwrap();
    //     assert_eq!(result, expected);
    // }

    #[test]
    fn test_parse_app() {
        let input = "foo(1, 2, 3)";
        let expected = Term::App(AppTerm::new("foo", vec![1.into(), 2.into(), 3.into()]));
        let (_, result) = parse_term(input).unwrap();
        assert_eq!(result, expected);

        let input = "bar(X, true, \"hello\")";
        let expected = Term::App(AppTerm::new(
            "bar",
            vec![
                Term::Var(Var::new("X")),
                Term::True,
                Term::Str("hello".to_string()),
            ],
        ));
        let (_, result) = parse_term(input).unwrap();
        assert_eq!(result, expected);
    }

    // #[test]
    // fn test_parse_complex_term() {
    //     let input = "foo(X, [1, 2, bar(Y)], {\"key\" => V})";
    //     let expected = Term::App(AppTerm::new(
    //         "foo",
    //         vec![
    //             Term::Var(Var::new("X")),
    //             Term::List(vec![
    //                 1.into(),
    //                 2.into(),
    //                 app!("bar", [Term::Var(Var::new("Y"))]),
    //             ]),
    //             Term::Map({
    //                 let mut map = BTreeMap::new();
    //                 map.insert(Term::Str("key".to_string()), Term::Var(Var::new("V")));
    //                 map
    //             }),
    //         ],
    //     ));
    //     let (_, result) = parse_term(input).unwrap();
    //     assert_eq!(result, expected);
    // }

    #[test]
    fn test_parse_rule_fact() {
        let input = "foo(1, 2).";
        let expected = Rule {
            head: AppTerm::new("foo", vec![1.into(), 2.into()]).into(),
            tail: vec![],
        };
        let (_, result) = parse_rule(input).unwrap();
        assert_eq!(result, expected);


        let input = "Foo.";
        let expected = Rule {
            head: term!(var("Foo")),
            tail: vec![],
        };
        let (_, result) = parse_rule(input).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_parse_rule_with_tail() {
        let input = "foo(X) :- bar(X), baz(Y).";
        let expected = Rule {
            head: AppTerm::new("foo", vec![Term::Var(Var::new("X"))]).into(),
            tail: vec![
                app!("bar", [Term::Var(Var::new("X"))]),
                app!("baz", [Term::Var(Var::new("Y"))]),
            ],
        };
        let (_, result) = parse_rule(input).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_parse_rule_with_multiple_goals() {
        let input = "ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).";
        let expected = Rule {
            head: AppTerm::new(
                "ancestor",
                vec![Term::Var(Var::new("X")), Term::Var(Var::new("Y"))],
            )
            .into(),
            tail: vec![
                app!(
                    "parent",
                    [Term::Var(Var::new("X")), Term::Var(Var::new("Z"))]
                ),
                app!(
                    "ancestor",
                    [Term::Var(Var::new("Z")), Term::Var(Var::new("Y"))]
                ),
            ],
        };
        let (_, result) = parse_rule(input).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_parse_query_single_goal() {
        let input = "?- foo(X).";
        let expected = Query::new(vec![app!("foo", [Term::Var(Var::new("X"))])]);
        let (_, result) = parse_query(input).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_parse_query_multiple_goals() {
        let input = "?- parent(X, Y), ancestor(Y, Z).";
        // let expected = Query {
        //     goals: vec![
        //         app!(
        //             "parent",
        //             [Term::Var(Var::new("X")), Term::Var(Var::new("Y"))]
        //         ),
        //         app!(
        //             "ancestor",
        //             [Term::Var(Var::new("Y")), Term::Var(Var::new("Z"))]
        //         ),
        //     ],
        // };
        let expected = Query::new(vec![
            app!("parent", [Term::Var(Var::new("X")), Term::Var(Var::new("Y"))]),
            app!("ancestor", [Term::Var(Var::new("Y")), Term::Var(Var::new("Z"))]),
        ]);

        let (_, result) = parse_query(input).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_parse_multiple_rules() {
        let input = "
            foo(1, 2).
            bar(X) :- baz(X), qux(Y).
            ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
        ";
        let expected = vec![
            Rule {
                head: AppTerm::new("foo", vec![1.into(), 2.into()]).into(),
                tail: vec![],
            },
            Rule {
                head: AppTerm::new("bar", vec![Term::Var(Var::new("X"))]).into(),
                tail: vec![
                    app!("baz", [Term::Var(Var::new("X"))]),
                    app!("qux", [Term::Var(Var::new("Y"))]),
                ],
            },
            Rule {
                head: AppTerm::new(
                    "ancestor",
                    vec![Term::Var(Var::new("X")), Term::Var(Var::new("Y"))],
                )
                .into(),
                tail: vec![
                    app!(
                        "parent",
                        [Term::Var(Var::new("X")), Term::Var(Var::new("Z"))]
                    ),
                    app!(
                        "ancestor",
                        [Term::Var(Var::new("Z")), Term::Var(Var::new("Y"))]
                    ),
                ],
            },
        ];
        let (_, result) = parse_rules(input).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_parse_invalid_rule() {
        let input = "42 :- foo(X)."; // Head is not an AppTerm
        let result = parse_rule(input);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_invalid_query() {
        let input = "?- foo(X), bar(Y)"; // Missing period
        let result = parse_query(input);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_empty_input() {
        let input = "";
        // Parsing rules should return empty
        let (_, rules) = parse_rules(input).unwrap();
        assert!(rules.is_empty());

        // Parsing a term should fail
        let result = parse_term(input);
        assert!(result.is_err());

        // Parsing a query should fail
        let result = parse_query(input);
        assert!(result.is_err());
    }

    // #[test]
    // fn test_parse_whitespace_handling() {
    //     let input = "  foo(  X , [1,  2, bar(Y) ], {\"key\" => V} )  ";
    //     let expected = Term::App(AppTerm::new(
    //         "foo",
    //         vec![
    //             Term::Var(Var::new("X")),
    //             Term::List(vec![
    //                 Term::Int(1),
    //                 Term::Int(2),
    //                 app!("bar", [Term::Var(Var::new("Y"))]),
    //             ]),
    //             Term::Map({
    //                 let mut map = BTreeMap::new();
    //                 map.insert(Term::Str("key".to_string()), Term::Var(Var::new("V")));
    //                 map
    //             }),
    //         ],
    //     ));
    //     let (_, result) = parse_term(input).unwrap();
    //     assert_eq!(result, expected);
    // }

    // #[test]
    // fn test_parse_nested_structures() {
    //     let input = "foo(bar(baz(X)), [1, {Y => true}, v[qux(Z)]]).";
    //     let expected = Rule {
    //         head: AppTerm::new(
    //             "foo",
    //             vec![
    //                 app!("bar", [app!("baz", [Term::Var(Var::new("X"))])]),
    //                 Term::List(vec![
    //                     Term::Int(1),
    //                     Term::Map({
    //                         let mut map = BTreeMap::new();
    //                         map.insert(Term::Var(Var::new("Y")), Term::True);
    //                         map
    //                     }),
    //                     Term::Union(vec![app!("qux", [Term::Var(Var::new("Z"))])]),
    //                 ]),
    //             ],
    //         )
    //         .into(),
    //         tail: vec![],
    //     };
    //     let (_, result) = parse_rule(input).unwrap();
    //     assert_eq!(result, expected);
    // }

    #[test]
    fn test_parse_rule_with_empty_tail() {
        let input = "foo(X) :- .";
        let expected = Rule {
            head: AppTerm::new("foo", vec![Term::Var(Var::new("X"))]).into(),
            tail: vec![], // Empty tail
        };
        let (_, result) = parse_rule(input).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_parse_rule_with_only_cut() {
        let input = "cut_rule :- !.";
        let expected = Rule {
            head: AppTerm::new("cut_rule", vec![]).into(),
            tail: vec![Term::Cut],
        };
        let (_, result) = parse_rule(input).unwrap();
        assert_eq!(result, expected);
    }
 */
}
