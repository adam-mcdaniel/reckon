pub mod solvers;
mod symbol;
use nom::error::VerboseError;
pub use solvers::Solver;

mod parse;
pub use parse::*;

mod env;
pub use env::*;

mod rule;
pub use rule::*;

mod query;
pub use query::*;


use std::sync::Arc;
use std::collections::{HashSet, HashMap, VecDeque};
use std::convert;
use std::fmt::{Debug, Display, Formatter, Result as FmtResult};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::str::FromStr;
use lazy_static::lazy_static;
use tracing::{debug, error, info, warn};

pub use symbol::Symbol;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Var {
    original_id: u64,
    id: u64,
}

impl Var {
    pub fn new(name: impl ToString) -> Self {
        // Create a symbol and get its ID
        let symbol = Symbol::new(name.to_string().as_str());
        let id = symbol.id();

        Var { original_id: id, id }
    }

    fn refresh(&self) -> Self {
        Var { id: Symbol::unused_id(), ..*self }
    }
}

impl FromStr for Var {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Var::from(s))
    }
}

impl<'a> From<&'a str> for Var {
    fn from(s: &'a str) -> Self {
        Var::new(s)
    }
}

/// Representation of a logical term
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Term {
    /// A non-variable symbol "x"
    Sym(Symbol),
    /// A variable term "?x"
    Var(Var),
    /// An application term
    App(AppTerm),
    /// A signed integer term
    Int(i64),
    /// A string term
    Str(String),
    /// A nil term
    Nil,
    /// Truth
    True,
    /// Falsehood
    False,

    Cons(Box<Term>, Box<Term>),
    Complement(Box<Term>),
    /// The special built-in cut predicate.
    ///
    /// Evaluating it prunes all further choices for the currently active rule.
    Cut,
}

impl Term {
    pub fn size(&self) -> usize {
        match self {
            Term::Sym(_) => 1,
            Term::Var(_) => 1,
            Term::App(app) => 1 + app.args.iter().map(|arg| arg.size()).sum::<usize>(),
            Term::Int(_) | Term::True | Term::False | Term::Str(_) | Term::Nil | Term::Cut => 1,
            Term::Cons(head, tail) => 1 + head.size() + tail.size(),
            Term::Complement(term) => 1 + term.size(),
        }
    }

    pub fn var(name: impl ToString) -> Self {
        Term::Var(Var::new(name))
    }

    pub fn app(func: impl ToString, args: Vec<Term>) -> Self {
        Term::App(AppTerm::new(func, args))
    }

    pub fn unify<S>(&self, other: &Self, env: &mut Env<S>, negated: bool) -> Result<Self, (Self, Self)> where S: Solver {
        let mut new = self.clone();
        new.unify_in_place(other, env, negated)?;
        new.substitute(&env.var_bindings());
        Ok(new)
    }

    pub fn negate(&self) -> Self {
        match self {
            Term::Complement(term) => *term.clone(),
            Term::True => Term::False,
            Term::False => Term::True,
            _ => Term::Complement(Box::new(self.clone())),
        }
        // Term::Complement(Box::new(self.clone()))
    }

    pub(self) fn unify_in_place<S>(
        &mut self,
        other: &Self,
        env: &mut Env<S>,
        negated: bool,
    ) -> Result<(), (Self, Self)> where S: Solver {
        let mut tmp_env = env.clone();
        if let Ok(()) = self.unify_in_place_helper(other, &mut tmp_env, negated) {
            *env = tmp_env;
            Ok(())
        } else {
            Err((self.clone(), other.clone()))
        }
    }

    pub(self) fn unify_in_place_helper<S>(
        &mut self,
        other: &Self,
        env: &mut Env<S>,
        negated: bool,
    ) -> Result<(), (Self, Self)> where S: Solver {
        use Term::*;

        if self == other && !negated {
            return Ok(());
        } else if self == other && negated {
            return Err((self.clone(), other.clone()));
        }
        
        let err = |self_, other_: &Self| {
            debug!("Unification error: {} != {}", self_, other_);
            Err((self_, other_.clone()))
        };
        debug!("Unifying {} with {}", self, other);

        match (self, other) {
            (Var(var1), Var(var2)) if !negated => {
                // If the variables are the same, they are already unified
                if var1 == var2 {
                    return Ok(());
                }

                // If the variable is already bound, unify the bound term with the other variable
                if let Some(mut term) = env.get_var(*var1).cloned() {
                    term.unify_in_place_helper(other, env, negated)?;
                } else if let Some(term) = env.get_var(*var2).cloned() {
                    // term.unify_in_place_helper(&Var(*var1), env)?;
                    Var(*var1).unify_in_place_helper(&term, env, negated)?;
                } else {
                    // Bind the variable to the other term
                    env.set_var(*var1, other.clone());
                }
            }
            (Var(var), term) => {
                if let Some(mut term2) = env.get_var(*var).cloned() {
                    term2.unify_in_place_helper(term, env, negated)?;
                } else if !negated {
                    env.set_var(*var, term.clone());
                }
            }
            (term, Var(var)) => {
                if let Some(term2) = env.get_var(*var).cloned() {
                    term.unify_in_place_helper(&term2, env, negated)?;
                } else if !negated {
                    env.set_var(*var, term.clone());
                }
            }

            (App(app1), App(app2)) if app1.func == app2.func && app1.args.len() == app2.args.len() => {
                for (arg1, arg2) in app1.args_mut().zip(app2.args.iter()) {
                    arg1.unify_in_place_helper(arg2, env, negated)?;
                }
            }

            (Complement(term1), Complement(term2)) => {
                term1.unify_in_place_helper(term2, env, negated)?;
            }
            (Complement(term), True) => {
                term.unify_in_place_helper(&False, env, negated)?;
            }
            (Complement(term), False) => {
                term.unify_in_place_helper(&True, env, negated)?;
            }
            (True, Complement(term)) => {
                term.unify(&False, env, negated)?;
            }
            (False, Complement(term)) => {
                term.unify(&True, env, negated)?;
            }
            (Complement(term), term2) => {
                if let Ok(term) = term.unify(term2, env, !negated) {
                    return Ok(());
                }
                return err(Complement(term.clone()), term2);
            }
            (term1, Complement(term)) => {
                if let Ok(term) = term1.unify(term, env, !negated) {
                    return Ok(());
                }
                return err(term1.clone(), &Complement(term.clone()));
                // if let Ok(term) = term1.unify(term, env) {
                //     return err(term1.clone(), &Complement(Box::new(term)));
                // } else {
                //     return Ok(());
                // }
            }

            (Int(n1), Int(n2)) if n1 == n2 && !negated => {
                info!("Unifying integers: {} == {}", n1, n2);
            }
            (Int(n1), Int(n2)) if n1 != n2 && negated => {
                info!("Unifying integers: {} == {}", n1, n2);
            }
            (True, True) | (False, False) if !negated => {
            }
            (False, True) | (True, False) if negated => {
            }
            (Str(s1), Str(s2)) if s1 == s2 => {}

            (Nil, Nil) | (Cut, Cut) => {}


            (Cons(head1, tail1), Cons(head2, tail2)) => {
                // head1.unify_in_place_helper(head2, bindings)?;
                // tail1.unify_in_place_helper(tail2, bindings)?;

                head1.unify_in_place_helper(head2, env, negated)?;
                tail1.unify_in_place_helper(tail2, env, negated)?;
            }

            (Cons(head, tail), term) => {
                // Match the head with the term, and the tail with nil
                head.unify_in_place_helper(term, env, negated)?;
                tail.unify_in_place_helper(&Nil, env, negated)?;
            }

            (term, Cons(head, tail)) => {
                term.unify_in_place_helper(head, env, negated)?;
                tail.unify(&Nil, env, negated)?;
            }

            (self_, _) => {
                debug!("Type mismatch: {} != {}", self_, other);
                return err(self_.clone(), other);
            }
        }
        Ok(())
    }

    fn has_used_vars(&self) -> bool {
        use Term::*;

        let mut has_found = false;
        self.traverse(&mut |term| {
            if let Var(_) = term {
                has_found = true;
            }
            // Continue if we haven't found a free variable yet
            !has_found
        });
        has_found
    }

    fn used_vars(&self, vars: &mut HashSet<Var>) {
        use Term::*;

        self.traverse(&mut |term| {
            if let Var(var) = term {
                vars.insert(*var);
            }
            // Continue for all terms
            true
        });

        // match self {
        //     Var(var) => {
        //         vars.insert(*var);
        //     }
        //     App(app) => {
        //         for arg in app.args.iter() {
        //             arg.used_vars(vars);
        //         }
        //     }
        //     Cons(head, tail) => {
        //         head.used_vars(vars);
        //         tail.used_vars(vars);
        //     }
        //     // Set(terms) => {
        //     //     for term in terms.iter() {
        //     //         term.used_vars(vars);
        //     //     }
        //     // }
        //     // Map(terms) => {
        //     //     for (key, val) in terms.iter() {
        //     //         key.used_vars(vars);
        //     //         val.used_vars(vars);
        //     //     }
        //     // }
        //     Complement(term) => {
        //         term.used_vars(vars);
        //     }
        //     Int(_) | Str(_) | Nil | Cut | True | False => {}
        // }
    }

    fn used_params(&self, params: &mut HashSet<Var>) {
        use Term::*;
        self.traverse(&mut |term| {
            if let App(app) = self {
                app.params(params);
            }
            true
        });
        // match self {
        //     Sym(_) => {}
        //     Var(_) => {}
        //     Complement(term) => term.used_params(params),
        //     App(app) => app.params(params),
        //     Cons(head, tail) => {
        //         head.used_params(params);
        //         tail.used_params(params);
        //     }
        //     // Set(terms) => {
        //     //     for term in terms.iter() {
        //     //         term.used_params(params);
        //     //     }
        //     // }
        //     // Map(terms) => {
        //     //     for (key, val) in terms.iter() {
        //     //         key.used_params(params);
        //     //         val.used_params(params);
        //     //     }
        //     // }
        //     Int(_) | Str(_) | Nil | Cut | True | False => {}
        // }
    }

    pub fn reduce<S>(&self, env: &Env<S>) -> Self where S: Solver {
        let mut term = self.clone();
        term.reduce_in_place(env);
        term
    }

    pub fn reduce_in_place<S>(&mut self, env: &Env<S>) where S: Solver {
        self.traverse_mut(&mut |term| {
            term.substitute(&env.var_bindings());

            match term {
                Term::Complement(term) => {
                    term.reduce_in_place(env);
                    match term.as_ref() {
                        Term::True => **term = Term::False,
                        Term::False => **term = Term::True,
                        _ => {}
                    }
                },
                _ => {}
            }

            true
        });

        // self.substitute(&env.var_bindings());

        // match self {
        //     Self::Complement(term) => {
        //         term.reduce_in_place(env);
        //         *self = match term.as_ref() {
        //             Self::True => Self::False,
        //             Self::False => Self::True,
        //             other => Self::Complement(Box::new(other.clone())),
        //         }
        //     },
        //     _ => {}
        // };

        // for child in self.subterms_mut() {
        //     child.reduce_in_place(env);
        // }
    }

    pub fn subterms(&self) -> Vec<&Self> {
        use Term::*;

        let mut subterms = vec![];
        match self {
            Complement(term) => {
                subterms.push(term.as_ref());
            }
            Sym(_) | Var(_) | Int(_) | Str(_) | Nil | Cut | True | False => {}
            App(app) => {
                subterms.extend(app.args.iter());
            }
            Cons(head, tail) => {
                subterms.push(head);
                subterms.push(tail);
            }
        }

        subterms
    }

    pub fn subterms_mut(&mut self) -> Vec<&mut Self> {
        use Term::*;

        let mut subterms: Vec<&mut Term> = vec![];
        match self {
            Complement(term) => {
                subterms.push(term);
            }
            Sym(_) | Var(_) | Int(_) | Str(_) | Nil | Cut | True | False => {}
            App(app) => {
                subterms.extend(app.args_mut());
            }
            Cons(head, tail) => {
                subterms.push(head);
                subterms.push(tail);
            }
        }

        subterms
    }

    pub fn traverse(&self, f: &mut impl FnMut(&Self) -> bool) {
        let mut queue = VecDeque::new();
        queue.push_back(self);

        while let Some(term) = queue.pop_front() {
            if !f(term) {
                return;
            }
            queue.extend(term.subterms());
        }
    }

    pub fn traverse_mut(&mut self, f: &mut impl FnMut(&mut Self) -> bool) {
        let mut queue = VecDeque::new();
        queue.push_back(self);

        while let Some(mut term) = queue.pop_front() {
            if !f(&mut term) {
                return;
            }
            queue.extend(term.subterms_mut());
        }
    }

    pub fn substitute(&mut self, bindings: &HashMap<Var, Term>) {
        use Term::*;

        if bindings.is_empty() {
            return;
        }

        self.traverse_mut(&mut |term| {
            if let Var(var) = term {
                if let Some(substitution) = bindings.get(var) {
                    *term = substitution.clone();
                }
            }
            true
        });

        // if let Var(var) = self {
        //     if let Some(term) = bindings.get(var) {
        //         *self = term.clone();
        //     }
        // } else {
        //     for child in self.subterms_mut() {
        //         child.substitute(bindings);
        //     }
        // }



        // match self {
        //     Complement(term) => {
        //         term.substitute(bindings);
        //     }
        //     Var(var) => {
        //         if let Some(term) = bindings.get(var) {
        //             *self = term.clone();
        //         }
        //     }
        //     App(app) => {
        //         for arg in app.args.iter_mut() {
        //             arg.substitute(bindings);
        //         }
        //     }
        //     Cons(head, tail) => {
        //         head.substitute(bindings);
        //         tail.substitute(bindings);
        //     }
        //     // Map(terms) => {
        //     //     *terms = terms
        //     //         .iter()
        //     //         .map(|(key, val)| {
        //     //             let mut key = key.clone();
        //     //             let mut val = val.clone();
        //     //             key.substitute(bindings);
        //     //             val.substitute(bindings);
        //     //             (key, val)
        //     //         })
        //     //         .collect();
        //     // }

        //     Cut | Nil | Int(_) | Str(_) | Sym(_) | True | False => {}
        // }
    }

    pub fn substitute_var(&mut self, var: Var, term: Term) {
        let bindings = [(var, term)].iter().cloned().collect();
        self.substitute(&bindings);
    }

    fn has_application_of(&self, app_term: &AppTerm) -> bool {
        // fn has_recursive_application(self_: &AppTerm, term: &Term) -> bool {
        //     match term {
        //         Term::App(app) => app.func == self_.func || term.subterms().iter().any(|term| has_recursive_application(self_, term)),
        //         _ => {
        //             term.subterms().iter().any(|term| has_recursive_application(self_, term))
        //         }
        //     }
        // };

        if let Term::App(self_) = self {
            if self_.func == app_term.func {
                return true;
            }

            return self_.args.iter().any(|term| term.has_application_of(app_term));
        } else {
            return self.subterms().iter().any(|term| term.has_application_of(app_term));
        }
    }
}

impl FromStr for Term {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parse_term(s)
            .map_err(|e| match e {
                nom::Err::Error(e) | nom::Err::Failure(e) => nom::error::convert_error(s, e),
                nom::Err::Incomplete(_) => "Incomplete input".to_string(),
            })
            .map(|(_, term)| term)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AppTerm {
    pub func: Symbol,
    pub args: Arc<Vec<Term>>,
}

impl AppTerm {
    pub fn new(func: impl ToString, args: Vec<Term>) -> Self {
        let func = Symbol::new(func.to_string().as_str());
        AppTerm { func, args: Arc::new(args) }
    }

    fn params(&self, params: &mut HashSet<Var>) {
        for arg in self.args.iter() {
            arg.used_vars(params);
        }
    }

    pub fn args(&self) -> impl Iterator<Item = &Term> {
        self.args.iter()
    }

    pub fn args_mut(&mut self) -> impl Iterator<Item = &mut Term> {
        Arc::make_mut(&mut self.args).iter_mut()
    }
}

pub fn var(name: impl ToString) -> Term {
    Term::var(name)
}

// pub fn app(func: impl ToString, args: Vec<Term>) -> Term {
//     Term::app(func, args)
// }
#[macro_export]
macro_rules! app {
    ($func:expr, [ $($arg:expr),* $(,)? ]) => {
        Term::App(AppTerm::new($func, vec![$(term!($arg)),*]))
    };

    ($func:expr, $arg:expr) => {
        Term::App(AppTerm::new($func, $arg))
    };
}

#[macro_export]
macro_rules! term {
    (cons $head:expr; $tail:expr) => {
        Term::Cons(Box::new(term!($head)), Box::new(term!($tail)))
    };

    // Match a literal expression
    ($val:expr) => {{
        Term::from($val)
    }};
}

impl From<AppTerm> for Term {
    fn from(a: AppTerm) -> Self {
        Term::App(a)
    }
}

impl From<i64> for Term {
    fn from(n: i64) -> Self {
        Term::Int(n)
    }
}

impl From<bool> for Term {
    fn from(b: bool) -> Self {
        if b {
            Term::True
        } else {
            Term::False
        }
    }
}

impl From<String> for Term {
    fn from(s: String) -> Self {
        Term::Str(s)
    }
}

impl From<Var> for Term {
    fn from(v: Var) -> Self {
        Term::Var(v)
    }
}

impl From<&str> for Term {
    fn from(s: &str) -> Self {
        Term::Str(s.to_string())
    }
}

impl From<Option<Term>> for Term {
    fn from(opt: Option<Term>) -> Self {
        match opt {
            Some(term) => term,
            None => Term::Nil,
        }
    }
}

impl Debug for Var {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{}", self)
    }
}

impl Display for Var {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        if self.original_id != self.id {
            write!(f, "{}-#{}", Symbol::from_id(self.original_id), self.id)
        } else {
            write!(f, "{}", Symbol::from_id(self.id))
        }
    }
}

impl Display for Term {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        match self {
            Term::Complement(term) => write!(f, "~{}", term),
            Term::Sym(sym) => write!(f, "{}", sym),
            Term::Var(var) => write!(f, "{}", var),
            Term::App(app) => write!(f, "{}", app),
            Term::Int(n) => write!(f, "{}", n),
            Term::True => write!(f, "true"),
            Term::False => write!(f, "false"),
            Term::Str(s) => write!(f, "{}", s),
            Term::Cons(head, tail) => {
                // write!(f, "[{:?}|{:?}]", head, tail)
                write!(f, "[{}", head)?;
                let mut tail = tail.as_ref();
                while let Term::Cons(head, x) = tail {
                    write!(f, ", {}", head)?;
                    tail = x;
                }
                if let Term::Nil = tail {
                    write!(f, "]")
                } else {
                    write!(f, ",{}]", tail)
                }
            }
            // Term::Set(terms) => {
            //     write!(f, "{{")?;
            //     for (i, term) in terms.iter().enumerate() {
            //         if i > 0 {
            //             write!(f, ", ")?;
            //         }
            //         write!(f, "{:?}", term)?;
            //     }
            //     write!(f, "}}")
            // }
            // Term::Map(terms) => {
            //     write!(f, "{{")?;
            //     for (i, (key, val)) in terms.iter().enumerate() {
            //         if i > 0 {
            //             write!(f, ", ")?;
            //         }
            //         write!(f, "{:?} => {:?}", key, val)?;
            //     }
            //     write!(f, "}}")
            // }
            Term::Nil => write!(f, "nil"),
            Term::Cut => write!(f, "#"),
        }
    }
}

impl Display for AppTerm {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{}(", self.func)?;
        for (i, arg) in self.args.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", arg)?;
        }
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_used_vars() {
        let term: Term = "f(X, Y, Z, test(A, B, C))".parse().unwrap();
        let mut vars = HashSet::new();
        term.used_vars(&mut vars);

        let expected = vec!["X", "Y", "Z", "A", "B", "C"]
            .iter()
            .map(|s| Var::from_str(s).unwrap())
            .collect::<HashSet<_>>();

        assert_eq!(vars, expected);
    }
/*

    #[test]
    fn test_term_macro() {
        // let term = term!({
        //     "foo" => 42,
        //     "bar" => true,
        //     "baz" => "qux",
        //     // var("test") => term!({"ing", 1}),
        //     None => term!([1, 2, "three"]),
        // });

        // println!("{:#?}", term);

        // let expected = Term::Map({
        //     let mut map = HashMap::new();
        //     map.insert(Term::Str("foo".to_string()), Term::Int(42));
        //     map.insert(Term::Str("bar".to_string()), Term::True);
        //     map.insert(Term::Str("baz".to_string()), Term::Str("qux".to_string()));
        //     map.insert(
        //         var("test"),
        //         Term::Set({
        //             let mut set = HashSet::new();
        //             set.insert(Term::Str("ing".to_string()));
        //             set.insert(Term::Int(1));
        //             set
        //         }),
        //     );
        //     map.insert(
        //         Term::Nil,
        //         Term::List(vec![
        //             Term::Int(1),
        //             Term::Int(2),
        //             Term::Str("three".to_string()),
        //         ]),
        //     );

        //     map
        // });

        // assert_eq!(term, expected);
    }

    /// Test the cons unification
    #[test]
    fn test_cons_unification() {
        let mut cons = term!(cons var("X"); var("Y"));

        let mut env = Env::default();
        let list = term!([1, 2]);

        cons.unify_in_place_helper(&list, &mut env).unwrap();

        assert_eq!(cons, term!(cons 1; 2));


        let mut cons = term!(
            cons var("X");
            term!(cons var("Y"); var("Z")));
        let list = term!([1, 2, 3]);

        cons.unify_in_place_helper(&list, &mut env).unwrap();

        assert_eq!(cons, term!(cons 1; term!(cons 2; 3)));

        let mut cons = term!(
            cons var("X");
            term!(cons var("Y"); var("X")));
        let list = term!([1, 2, 1]);

        cons.unify_in_place_helper(&list, &mut env).unwrap();

        assert_eq!(cons, term!(cons 1; term!(cons 2; 1)));
    }

    /// Test unification
    #[test]
    fn test_unification() {
        let mut term1 = app!("foo", [var("X"), var("Y")]);
        let term2 = app!("foo", [42, true]);
        let mut env = Env::default();

        let mut ground_truth = HashMap::new();
        ground_truth.insert("X".into(), 42.into());
        ground_truth.insert("Y".into(), true.into());
        term1.unify_in_place_helper(&term2, &mut env).unwrap();
        assert_eq!(env.bindings, ground_truth);

        let mut term1 = app!("foo", [var("X"), var("Y")]);
        let term2 = app!("foo", [42, var("Z")]);
        term1.unify_in_place_helper(&term2, &mut env).unwrap();

        assert_eq!(term1, app!("foo", [42, var("Z")]));

        // Test unions
        let mut term1 = term!(v [
            app!("foo", [var("X"), var("Y")]),
            app!("bar", [var("Z")]),
        ]);

        let term2 = term!(v [
            app!("foo", [42, true]),
            app!("bar", [var("Z")]),
        ]);

        term1.unify_in_place_helper(&term2, &mut env).unwrap();

        assert_eq!(
            term1,
            term!(v [
                app!("foo", [42, true]),
                app!("bar", [var("Z")]),
            ])
        );

        // Test intersections
        let mut term1 = term!(^ [
            app!("foo", [var("X"), var("Y")]),
            app!("bar", [var("Z")]),
        ]);

        let term2 = term!(^ [
            app!("foo", [42, true]),
            app!("bar", [var("Z")]),
        ]);

        term1.unify_in_place_helper(&term2, &mut env).unwrap();

        assert_eq!(
            term1,
            term!(^ [
                app!("foo", [42, true]),
                app!("bar", [var("Z")]),
            ])
        );
    }

    /// Test successful unification of union expressions with multiple possible matches.
    #[test]
    fn test_union_unification_success() {
        let mut env = Env::default();
        // Union: foo(X, Y) ∪ bar(Z)
        let mut term1 = term!(v [
            app!("foo", [var("X"), var("Y")]),
            app!("bar", [var("Z")]),
        ]);

        // Union: foo(42, true) ∪ bar(W)
        let term2 = term!(v [
            app!("foo", [42, true]),
            app!("bar", [var("W")]),
        ]);

        // Perform unification
        term1.unify_in_place_helper(&term2, &mut env).unwrap();

        // After unification, term1 should have:
        // foo(42, true) ∪ bar(W)
        assert_eq!(
            term1,
            term!(v [
                app!("foo", [42, true]),
                app!("bar", [var("W")]),
            ])
        );
    }

    /// Test failed unification of union expressions when no operands can be unified.
    #[test]
    fn test_union_unification_failure() {
        let mut env = Env::default();
        
        // Union: foo(X) ∪ bar(Y)
        let mut term1 = term!(v [
            app!("foo", [var("X")]),
            app!("bar", [var("Y")]),
        ]);

        // Union: baz(1) ∪ qux(2)
        let term2 = term!(v [
            app!("baz", [1]),
            app!("qux", [2]),
        ]);

        // Perform unification, which should fail
        assert!(term1.unify_in_place_helper(&term2, &mut env).is_err());
    }

    /// Test successful unification of intersection expressions requiring all operands to match.
    #[test]
    fn test_intersection_unification_success() {
        let mut env = Env::default();

        // Intersection: foo(X, Y) ∩ bar(Z)
        let mut term1 = term!(^ [
            app!("foo", [var("X"), var("Y")]),
            app!("bar", [var("Z")]),
        ]);

        // Intersection: foo(42, true) ∩ bar(W)
        let term2 = term!(^ [
            app!("foo", [42, true]),
            app!("bar", [var("W")]),
        ]);

        // Perform unification
        term1.unify_in_place_helper(&term2, &mut env).unwrap();

        // After unification, term1 should have:
        // foo(42, true) ∩ bar(W)
        assert_eq!(
            term1,
            term!(^ [
                app!("foo", [42, true]),
                app!("bar", [var("W")]),
            ])
        );
    }

    /// Test failed unification of intersection expressions when one operand cannot be unified.
    #[test]
    fn test_intersection_unification_failure() {
        let mut env = Env::default();

        // Intersection: foo(X) ∩ bar(Y)
        let mut term1 = term!(^ [
            app!("foo", [var("X")]),
            app!("bar", [var("Y")]),
        ]);

        // Intersection: foo(1) ∩ baz(2)
        let term2 = term!(^ [
            app!("foo", [1]),
            app!("baz", [2]),
        ]);

        // Perform unification, which should fail due to "bar" vs "baz"
        assert!(term1.unify_in_place_helper(&term2, &mut env).is_err());
    }
    */
}
