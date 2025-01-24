use std::collections::{BTreeMap, BTreeSet};

use super::*;

use nom::{
    branch::alt,
    bytes::complete::{escaped, is_not, tag, take_while, take_while1, take_while_m_n},
    character::complete::{alphanumeric1, char, multispace0, multispace1, one_of},
    combinator::{cut, map, map_opt, map_res, opt, value, verify},
    error::{FromExternalError, ParseError},
    multi::{fold_many0, fold_many1, fold_many_m_n, separated_list0, separated_list1},
    sequence::{delimited, preceded, separated_pair, terminated, tuple},
    IResult, Parser,
};

use crate::{
    var, // helper for building Var-based Terms
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
fn parse_int(i: &str) -> IResult<&str, Term, Error> {
    // 1. Optional sign
    // 2. Some digits
    let (i, num_str) = ws(map_res(recognize_sign_digits, |s: &str| s.parse::<i64>()))(i)?;
    Ok((i, Term::Int(num_str)))
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
fn parse_bool(i: &str) -> IResult<&str, Term, Error> {
    alt((
        value(Term::True, ws(tag("true"))),
        value(Term::False, ws(tag("false"))),
    ))(i)
}

/// Parse `nil`.
fn parse_nil(i: &str) -> IResult<&str, Term, Error> {
    value(Term::Nil, ws(tag("nil")))(i)
}

/// Parse the cut symbol `#`.
fn parse_cut(i: &str) -> IResult<&str, Term, Error> {
    value(Term::Cut, ws(tag("#")))(i)
}

/// `tag(string)` generates a parser that recognizes the argument string.
///
/// we can combine it with other functions, like `value` that takes another
/// parser, and if that parser returns without an error, returns a given
/// constant value.
///
/// `alt` is another combinator that tries multiple parsers one by one, until
/// one of them succeeds
fn boolean<'a, E: ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, bool, E> {
    // This is a parser that returns `true` if it sees the string "true", and
    // an error otherwise
    let parse_true = value(true, tag("true"));

    // This is a parser that returns `false` if it sees the string "false", and
    // an error otherwise
    let parse_false = value(false, tag("false"));

    // `alt` combines the two parsers. It returns the result of the first
    // successful parser, or an error
    alt((parse_true, parse_false)).parse(input)
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
    let (i, sym) = ws(map_res(
        take_while1(|c: char| c.is_ascii_alphanumeric() || c == '_'),
        |s: &str| {
            let first_char = s.chars().next().unwrap();
            if first_char.is_lowercase() {
                Ok(s.to_string())
            } else {
                Err("Not a symbol")
            }
        },
    ))(i)?;
    Ok((i, sym))
}

// --- Structured term parsers ---

/// Parse a list: `[ term, term, ... ]`
fn parse_list(i: &str) -> IResult<&str, Term, Error> {
    let (i, terms) = delimited(
        ws(char('[')),
        separated_list0(ws(char(',')), parse_term),
        ws(char(']')),
    )(i)?;
    Ok((i, Term::List(terms)))
}

/// Parse a cons cell: `[term | term]`
/// This is a special case of a list, where the tail is a single term.
fn parse_cons(i: &str) -> IResult<&str, Term, Error> {
    let (i, _) = ws(char('['))(i)?;
    let (i, (head, _, tail)) = tuple((ws(parse_term), ws(char('|')), ws(parse_term)))(i)?;
    let (i, _) = ws(char(']'))(i)?;
    Ok((i, Term::Cons(Box::new(head), Box::new(tail))))
}

/// Parse a set: `{ term, term, ... }`
// fn parse_set(i: &str) -> IResult<&str, Term, Error> {
//     let (i, terms) = delimited(
//         ws(char('{')),
//         separated_list0(ws(char(',')), parse_term),
//         ws(char('}')),
//     )(i)?;

//     let mut set = BTreeSet::new();
//     for t in terms {
//         set.insert(t);
//     }
//     Ok((i, Term::Set(set)))
// }

// /// Parse a map: `{ key => value, key => value, ... }`
// fn parse_map(i: &str) -> IResult<&str, Term, Error> {
//     let parse_entry = separated_pair(parse_term, ws(tag("=>")), parse_term);

//     let (i, entries) = delimited(
//         ws(char('{')),
//         separated_list0(ws(char(',')), parse_entry),
//         ws(char('}')),
//     )(i)?;

//     let mut map = BTreeMap::new();
//     for (k, v) in entries {
//         map.insert(k, v);
//     }
//     Ok((i, Term::Map(map)))
// }

/// Parse a union: `v[ term, term, ... ]`
fn parse_union(i: &str) -> IResult<&str, Term, Error> {
    let (i, _) = ws(tag("v"))(i)?;
    let (i, terms) = delimited(
        ws(char('[')),
        separated_list0(ws(char(',')), parse_term),
        ws(char(']')),
    )(i)?;
    Ok((i, Term::Union(terms)))
}

/// Parse an intersection: `^[ term, term, ... ]`
fn parse_intersection(i: &str) -> IResult<&str, Term, Error> {
    let (i, _) = ws(tag("^"))(i)?;
    let (i, terms) = delimited(
        ws(char('[')),
        separated_list0(ws(char(',')), parse_term),
        ws(char(']')),
    )(i)?;
    Ok((i, Term::Intersect(terms)))
}

/// Parse an application term: `foo(term, term, ...)`
/// We parse the symbol first, then arguments in parentheses.
fn parse_app(i: &str) -> IResult<&str, Term, Error> {
    // parse the function name
    let (i, func_str) = parse_symbol(i)?;
    let (i, args) = delimited(
        ws(char('(')),
        separated_list0(ws(char(',')), parse_term),
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
        parse_bool,
        parse_int,
        map(parse_string, Term::Str),
        parse_var,
        parse_union,
        parse_intersection,
        parse_cons,
        parse_list,
        // parse_set,
        // parse_map,
    ))(i)
}

fn parse_equal(i: &str) -> IResult<&str, Term, Error> {
    let (i, (lhs, _, rhs)) = tuple((parse_term_atom, ws(tag("=")), parse_term_atom))(i)?;
    Ok((i, Term::Equal(Box::new(lhs), Box::new(rhs))))
}

fn parse_not_equal(i: &str) -> IResult<&str, Term, Error> {
    let (i, (lhs, _, rhs)) = tuple((parse_term_atom, ws(tag("!=")), parse_term_atom))(i)?;
    Ok((i, Term::NotEqual(Box::new(lhs), Box::new(rhs))))
}

fn parse_complement(i: &str) -> IResult<&str, Term, Error> {
    let (i, _) = ws(tag("~"))(i)?;
    let (i, term) = parse_term_atom(i)?;
    Ok((i, Term::Complement(Box::new(term))))
}

pub fn parse_term(i: &str) -> IResult<&str, Term, Error> {
    alt((parse_equal, parse_not_equal, parse_complement, parse_term_atom))(i)
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
pub fn parse_rule(i: &str) -> IResult<&str, Rule, Error> {
    alt((parse_app_rule, cut(parse_term_rule)))(i)
}

// --- Query parsing ---
//
// A query might be: `?- goal1, goal2, ... .`
// We'll parse a question mark, dash, then a list of terms separated by commas, ended by a period.

pub fn parse_query(i: &str) -> IResult<&str, Query, Error> {
    let (i, _) = ws(tag("?-"))(i)?;
    let (i, goals) = separated_list1(ws(char(',')), parse_term)(i)?;
    let (i, _) = ws(char('.'))(i)?;
    Ok((i, Query { goals }))
}

// --- Helper for parsing multiple rules or queries from a file/string ---

/// Parse multiple rules separated by whitespace until EOF.
pub fn parse_rules(i: &str) -> IResult<&str, Vec<Rule>, Error> {
    let mut acc = Vec::new();
    let mut input = i;

    loop {
        // Attempt to parse one rule
        let res = parse_rule(input);
        match res {
            Ok((rest, rule)) => {
                acc.push(rule);
                input = rest;
                // Continue if there's more
                if rest.trim().is_empty() {
                    break;
                }
            }
            // If we fail because there's no more data or partial leftover, stop.
            Err(_) => break,
        }
    }

    // Return the successfully parsed rules plus the remainder
    Ok((input, acc))
}

#[cfg(test)]
mod test {
    use super::*;

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

    #[test]
    fn test_parse_list() {
        let input = "[1, 2, 3]";
        let expected = Term::List(vec![1.into(), 2.into(), 3.into()]);
        let (_, result) = parse_term(input).unwrap();
        assert_eq!(result, expected);

        let input = "[X, true, \"hello\"]";
        let expected = Term::List(vec![
            Term::Var(Var::new("X")),
            Term::True,
            Term::Str("hello".to_string()),
        ]);
        let (_, result) = parse_term(input).unwrap();
        assert_eq!(result, expected);
    }

    // #[test]
    // fn test_parse_set() {
    //     let input = "{1, 2, 3}";
    //     let mut set = BTreeSet::new();
    //     set.insert(Term::Int(1));
    //     set.insert(Term::Int(2));
    //     set.insert(Term::Int(3));
    //     let expected = Term::Set(set);
    //     let (_, result) = parse_term(input).unwrap();
    //     assert_eq!(result, expected);

    //     let input = "{X, true, \"hello\"}";
    //     let mut set = BTreeSet::new();
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
    fn test_parse_union() {
        let input = "v[foo(1), bar(2)]";
        let expected = Term::Union(vec![app!("foo", [1]), app!("bar", [2])]);
        let (_, result) = parse_term(input).unwrap();
        assert_eq!(result, expected);

        let input = "v[ X, Y, Z ]";
        let expected = Term::Union(vec![
            Term::Var(Var::new("X")),
            Term::Var(Var::new("Y")),
            Term::Var(Var::new("Z")),
        ]);
        let (_, result) = parse_term(input).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_parse_intersection() {
        let input = "^[foo(1), bar(2)]";
        let expected = Term::Intersect(vec![app!("foo", [1]), app!("bar", [2])]);
        let (_, result) = parse_term(input).unwrap();
        assert_eq!(result, expected);

        let input = "^[ X, Y, Z ]";
        let expected = Term::Intersect(vec![
            Term::Var(Var::new("X")),
            Term::Var(Var::new("Y")),
            Term::Var(Var::new("Z")),
        ]);
        let (_, result) = parse_term(input).unwrap();
        assert_eq!(result, expected);
    }

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
        let expected = Query {
            goals: vec![app!("foo", [Term::Var(Var::new("X"))])],
        };
        let (_, result) = parse_query(input).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_parse_query_multiple_goals() {
        let input = "?- parent(X, Y), ancestor(Y, Z).";
        let expected = Query {
            goals: vec![
                app!(
                    "parent",
                    [Term::Var(Var::new("X")), Term::Var(Var::new("Y"))]
                ),
                app!(
                    "ancestor",
                    [Term::Var(Var::new("Y")), Term::Var(Var::new("Z"))]
                ),
            ],
        };
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
}
