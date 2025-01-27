
use std::sync::Arc;
use std::collections::{HashSet, HashMap, VecDeque};
use std::fmt::{Debug, Display, Formatter, Result as FmtResult};
use std::hash::Hash;
use std::str::FromStr;
use tracing::{debug, info, warn};

use super::*;
use crate::Symbol;

/// A variable in a logical expression -- an unknown term.
/// 
/// Variables can be bound to other terms, and are used to represent unknown values in logical rules.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Var {
    /// Variables are given an "original" ID based on their name upon creation.
    /// 
    /// Whenever they're refreshed, they get a new, unique ID, but the original ID is kept.
    /// 
    /// This is used for finding the original variable name when printing the variable.
    original_id: u64,

    /// This is the unique ID of the variable.
    /// 
    /// This ID is unique for each variable, and is used to identify the variable in the environment.
    /// 
    /// A term can be "refreshed" by creating a new term with all of the variables refreshed.
    /// This allows the solver to not have to worry about variable name clashes between different rules.
    id: u64,
}

impl Var {
    /// Create a new variable with a given name.
    /// 
    /// Two variables with the same name will have the same ID until refreshed.
    /// 
    /// Two variables with the same name, when both are refreshed, will have different IDs,
    /// so name clashes are avoided.
    pub fn new(name: impl ToString) -> Self {
        // Create a symbol and get its ID
        let symbol = Symbol::new(name.to_string().as_str());
        let id = symbol.id();

        Var { original_id: id, id }
    }

    /// Get a new unique variable that has not been used before.
    pub fn refresh(&self) -> Self {
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

/// Representation of a logical term.
/// 
/// A term is a tree structure, where each node is a term.
/// 
/// There are literal terms, like integers, strings, and booleans.
/// These are leaf nodes in the term tree.
/// 
/// There are also compound terms, like application terms and cons cells.
/// These are nodes with children in the term tree, and are used to represent more complex structures
/// which can be used in logical rules.
/// 
/// Variables are used to represent yet unknown terms, and can be bound to other terms.
/// Symbols are used to represent non-variable terms in logical expressions, but
/// with a unique, string interned identifier.
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

    /// A cons cell
    /// 
    /// This is a list cell with a head and a tail.
    Cons(Box<Term>, Box<Term>),

    /// A complement term
    /// 
    /// This is a term that is negated.
    Complement(Box<Term>),

    /// The special built-in cut predicate.
    ///
    /// Evaluating it prunes all further choices for the currently active rule.
    Cut,
}

impl Term {
    /// Get the size of this term.
    /// 
    /// The size of a term is the number of nodes in the term tree.
    /// 
    /// This is useful for creating a heuristic for selecting which terms to unify first.
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

    /// Create a new symbol term.
    pub fn var(name: impl ToString) -> Self {
        Term::Var(Var::new(name))
    }

    /// Create a new application term.
    pub fn app(func: impl ToString, args: Vec<Term>) -> Self {
        Term::App(AppTerm::new(func, args))
    }

    /// Complement this term.
    /// 
    /// If this term is a boolean term, it will be negated.
    /// If this term is already a complement, the complement will be removed.
    /// All other terms will be wrapped in a complement.
    pub fn negate(&self) -> Self {
        match self {
            Term::Complement(term) => *term.clone(),
            Term::True => Term::False,
            Term::False => Term::True,
            _ => {
                if !self.is_app() {
                    warn!("Negating non-application term {}, this could result in explosion💣!", self);
                }

                Term::Complement(Box::new(self.clone()))
            },
        }
        // Term::Complement(Box::new(self.clone()))
    }

    /// Unify this term with another term.
    /// 
    /// This will attempt to bind the variable terms in this term to the corresponding terms in the other term.
    /// 
    /// If the structures of the terms are incompatible, an error is returned.
    /// 
    /// The `self` term will not be modified, and a new term will be returned.
    /// For in-place unification, use `unify_in_place`.
    pub fn unify<S>(&self, other: &Self, env: &mut Env<S>, negated: bool) -> Result<Self, (Self, Self)> where S: Solver {
        let mut new = self.clone();
        new.unify_in_place(other, env, negated)?;
        new.substitute(&env.var_bindings());
        Ok(new)
    }

    /// Unify this term with another term.
    /// 
    /// This will attempt to bind the variable terms in this term to the corresponding terms in the other term.
    /// 
    /// If the structures of the terms are incompatible, an error is returned, and the environment is not modified.
    pub fn unify_in_place<S>(
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

    /// Is this term an application term?
    pub fn is_app(&self) -> bool {
        matches!(self, Term::App(_))
    }

    /// A helper function for unifying two terms.
    /// 
    /// It will attempt to unify the two terms, and return an error if they cannot be unified.
    /// However, it modifies the environment as it goes, so it should not be used directly.
    /// 
    /// If the terms can't be unified, there still might be some changes to the environment,
    /// which should then be ignored.
    /// 
    /// This function is used by `unify_in_place` and `unify`.
    fn unify_in_place_helper<S>(
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
            (Sym(sym1), Sym(sym2)) if sym1 == sym2 && !negated => {}

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
                } else if negated {
                    // Unify the complement of the variables to each other
                    let mut complement1 = Complement(Box::new(Var(*var1)));
                    let mut complement2 = Complement(Box::new(Var(*var2)));

                    complement1.unify_in_place_helper(&Var(*var2), env, negated)?;
                    complement2.unify_in_place_helper(&Var(*var1), env, negated)?;
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
            // (Complement(term), term2) => {
            //     if let Ok(()) = term.unify_in_place(term2, env, !negated) {
            //         return Ok(());
            //     }
            //     return err(Complement(term.clone()), term2);
            // }
            // (term1, Complement(term)) => {
            //     if let Ok(()) = term1.unify_in_place(term, env, !negated) {
            //         return Ok(());
            //     }
            //     return err(term1.clone(), &Complement(term.clone()));
            //     // if let Ok(term) = term1.unify(term, env) {
            //     //     return err(term1.clone(), &Complement(Box::new(term)));
            //     // } else {
            //     //     return Ok(());
            //     // }
            // }

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

    /// Does this term contain any variables?
    pub fn has_vars(&self) -> bool {
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

    /// Get the set of variables used in this term.
    /// 
    /// The variables used in this term are collected and inserted into the given set.
    pub fn used_vars(&self, vars: &mut HashSet<Var>) {
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

    /// Return a new, reduced term.
    /// 
    /// This will apply the environment's variable bindings to this term, performing
    /// variable substitution and simplification.
    /// 
    /// Complements of boolean terms will be simplified to their negation.
    pub fn reduce<S>(&self, env: &Env<S>) -> Self where S: Solver {
        let mut term = self.clone();
        term.reduce_in_place(env);
        term
    }

    /// Reduce this term in place.
    /// 
    /// This will apply the environment's variable bindings to this term, performing
    /// variable substitution and simplification.
    /// 
    /// Complements of boolean terms will be simplified to their negation.
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
    }

    /// Get references to the subterms of this term.
    /// This will only return references to the subterms of this term, not the term itself.
    /// Also, it only returns references to the immediate subterms, not the subterms of subterms.
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

    /// Get mutable references to the subterms of this term.
    /// This will only return mutable references to the subterms of this term, not the term itself.
    /// Also, it only returns references to the immediate subterms, not the subterms of subterms.
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

    /// Apply a function that inspects this term, and all subterms recursively.
    /// 
    /// When the function returns false, the traversal stops.
    /// When the function returns true, the traversal continues.
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

    /// Apply a function that mutates this term, and all subterms recursively.
    /// 
    /// When the function returns false, the traversal stops.
    /// When the function returns true, the traversal continues.
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

    /// Substitute all occurrences of variables in this term with the given bindings.
    /// 
    /// This will modify this term in place, replacing all occurrences of variables with their corresponding terms in the bindings.
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
    }

    /// Substitute a single variable with a term.
    /// 
    /// This will modify this term in place, replacing all occurrences of the given variable with the given term.
    pub fn substitute_var(&mut self, var: Var, term: Term) {
        let bindings = [(var, term)].iter().cloned().collect();
        self.substitute(&bindings);
    }

    /// Does this term contain an application of the given application term?
    /// 
    /// If the term contains an application of a function with the same name as the given application term,
    /// this function returns true.
    /// 
    /// Otherwise, it returns false.
    pub fn has_application_of(&self, app_term: &AppTerm) -> bool {
        let mut found_app = false;
        self.traverse(&mut |term| {
            if let Term::App(app) = term {
                if app.func == app_term.func {
                    found_app = true;
                    return false;
                }
            }
            true
        });
        found_app
    }
}

/// Parse a term from a string.
/// 
/// This will use the `nom` parser combinator library to parse a term from a string.
/// 
/// Terms use a variant of the Prolog syntax, with some modifications.
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

/// A compound term in a logical expression.
/// 
/// This is an application term, which is a function applied to a list of arguments.
/// 
/// The function name is a symbol, and the arguments are logical terms.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AppTerm {
    /// The name of the function to apply
    pub func: Symbol,
    /// The terms to apply the function to
    pub args: Arc<Vec<Term>>,
}

impl AppTerm {
    /// Create a new application term.
    pub fn new(func: impl ToString, args: Vec<Term>) -> Self {
        let func = Symbol::new(func.to_string().as_str());
        AppTerm { func, args: Arc::new(args) }
    }

    /// Get the arguments of the application term as an iterator
    pub fn args(&self) -> impl Iterator<Item = &Term> {
        self.args.iter()
    }
    
    /// Get mutable references to the arguments of the application term as an iterator
    pub fn args_mut(&mut self) -> impl Iterator<Item = &mut Term> {
        Arc::make_mut(&mut self.args).iter_mut()
    }
}

/// A function to create a new variable term from a name.
pub fn var(name: impl ToString) -> Term {
    Term::var(name)
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
                write!(f, "[{}|{}]", head, tail)
                // write!(f, "[{}", head)?;
                // let mut tail = tail.as_ref();
                // while let Term::Cons(head, x) = tail {
                //     write!(f, ", {}", head)?;
                //     tail = x;
                // }
                // if let Term::Nil = tail {
                //     write!(f, "]")
                // } else {
                //     write!(f, ",{}]", tail)
                // }
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