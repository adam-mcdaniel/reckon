pub mod solvers;
mod symbol;
use nom::error::VerboseError;
pub use solvers::Solver;

mod parse;
pub use parse::*;


use std::sync::Arc;
use std::collections::{BTreeMap, BTreeSet};
use std::convert;
use std::fmt::{Debug, Display, Formatter, Result as FmtResult};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::str::FromStr;

use tracing::{debug, error, info, warn};

pub use symbol::Symbol;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Var {
    id: u64,
}

impl Var {
    pub fn new(name: impl ToString) -> Self {
        // Create a symbol and get its ID
        let symbol = Symbol::new(name.to_string().as_str());
        let id = symbol.id();
        Var { id }
    }

    fn refresh(&self) -> Self {
        Self::new(Symbol::from_id(self.id).refresh())
    }
}

impl FromStr for Var {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let symbol = Symbol::from_str(s).map_err(|_| ())?;
        Ok(Var { id: symbol.id() })
    }
}

impl<'a> From<&'a str> for Var {
    fn from(s: &'a str) -> Self {
        Var::from_str(s).unwrap()
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnifyEnv {
    rules: Arc<Vec<Rule>>,
    queries: Arc<Vec<Query>>,
    bindings: BTreeMap<Var, Term>,
}

impl UnifyEnv {
    pub fn new(rules: &[Rule], queries: &[Query]) -> Self {
        UnifyEnv {
            rules: Arc::new(rules.to_vec()),
            queries: Arc::new(queries.to_vec()),
            bindings: BTreeMap::new(),
        }
    }

    pub fn set_var(&mut self, var: Var, term: Term) {
        self.bindings.insert(var, term);
    }

    pub fn get_var(&self, var: Var) -> Option<&Term> {
        self.bindings.get(&var)
    }

    pub fn get_rules(&self) -> &[Rule] {
        &self.rules
    }

    pub fn add_rule(&mut self, rule: Rule) {
        Arc::make_mut(&mut self.rules).push(rule);
    }

    pub fn to_full_solution(&self, query: &Query) -> Result<Solution, BTreeSet<Var>> {
        let mut free_query_vars = BTreeSet::new();
        query.free_vars(&mut free_query_vars);

        // Now that we have the free variables from the query, 
        // simplify them in the bindings until there are no more free variables
        let mut bindings = self.bindings.clone();
        let mut has_found_free_vars = false;
        for _ in 0..100 {
            let previous_bindings = bindings.clone();
            for free_query_var in free_query_vars.clone() {
                if let Some(term) = bindings.get_mut(&free_query_var) {
                    if term.has_free_vars() {
                        term.substitute(&previous_bindings);
                        term.reduce_in_place(self);
                        has_found_free_vars = true;
                    } else {
                        free_query_vars.remove(&free_query_var);
                    }
                }
            }

            if !has_found_free_vars {
                break;
            }
        }

        if !free_query_vars.is_empty() {
            return Err(free_query_vars);
        }

        // Filter out the free variables that are not in the query
        let mut free_query_vars = BTreeSet::new();
        query.free_vars(&mut free_query_vars);
        bindings.retain(|var, _| free_query_vars.contains(var));

        Ok(Solution::new(query.clone(), bindings))
    }

    pub fn to_partial_solution(&self, query: &Query) -> Solution {
        let mut free_query_vars = BTreeSet::new();
        query.free_vars(&mut free_query_vars);

        // Now that we have the free variables from the query, 
        // simplify them in the bindings until there are no more free variables
        let mut bindings = self.bindings.clone();
        let mut has_found_free_vars = false;
        for _ in 0..100 {
            let previous_bindings = bindings.clone();
            for free_query_var in free_query_vars.clone() {
                if let Some(term) = bindings.get_mut(&free_query_var) {
                    if term.has_free_vars() {
                        term.substitute(&previous_bindings);
                        term.reduce_in_place(self);
                        has_found_free_vars = true;
                    } else {
                        free_query_vars.remove(&free_query_var);
                    }
                }
            }

            if !has_found_free_vars {
                break;
            }
        }
        
        Solution::new(query.clone(), bindings)
    }
}


#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Solution {
    pub query: Query,
    pub bindings: BTreeMap<Var, Term>,
}

impl Solution {
    pub fn new(query: Query, bindings: BTreeMap<Var, Term>) -> Self {
        Solution { query, bindings }
    }
}

impl From<Solution> for UnifyEnv {
    fn from(solution: Solution) -> Self {
        UnifyEnv {
            rules: Arc::new(vec![]),
            bindings: solution.bindings,
            queries: Arc::new(vec![solution.query]),
        }
    }
}

/// Representation of a logical term
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Term {
    /// A non-variable symbol
    Sym(Symbol),
    /// A variable term
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

    Equal(Box<Term>, Box<Term>),
    NotEqual(Box<Term>, Box<Term>),

    /// A list of terms
    List(Vec<Term>),
    /// A cons of terms
    Cons(Box<Term>, Box<Term>),
    // /// A set of terms
    // Set(BTreeSet<Term>),
    /// A map of terms
    // Map(BTreeMap<Term, Term>),
    // Idx(Box<Term>, Box<Term>),
    Union(Vec<Term>),
    Intersect(Vec<Term>),
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
            Term::Equal(lhs, rhs) | Term::NotEqual(lhs, rhs) => 1 + lhs.size() + rhs.size(),
            Term::App(app) => 1 + app.args.iter().map(|arg| arg.size()).sum::<usize>(),
            Term::Int(_) | Term::True | Term::False | Term::Str(_) | Term::Nil | Term::Cut => 1,
            Term::List(terms) => 1 + terms.iter().map(|term| term.size()).sum::<usize>(),
            Term::Cons(head, tail) => 1 + head.size() + tail.size(),
            Term::Complement(term) => 1 + term.size(),
            // Term::Set(terms) => 1 + terms.iter().map(|term| term.size()).sum::<usize>(),
            // Term::Map(terms) => 1 + terms.iter().map(|(key, val)| key.size() + val.size()).sum::<usize>(),
            Term::Union(terms) => 1 + terms.iter().map(|term| term.size()).sum::<usize>(),
            Term::Intersect(terms) => 1 + terms.iter().map(|term| term.size()).sum::<usize>(),
        }
    }

    pub fn var(name: impl ToString) -> Self {
        Term::Var(Var::new(name))
    }

    pub fn app(func: impl ToString, args: Vec<Term>) -> Self {
        Term::App(AppTerm::new(func, args))
    }

    pub fn unify(&self, other: &Self, env: &mut UnifyEnv) -> Result<Self, (Self, Self)> {
        let mut new = self.clone();
        new.unify_in_place(other, env)?;
        new.substitute(&env.bindings);
        Ok(new)
    }

    pub fn union(&mut self, other: Self) -> &mut Self {
        match self {
            Term::Union(terms) => {
                terms.push(other);
                self
            }
            _ => {
                let terms = vec![self.clone(), other];
                *self = Term::Union(terms);
                self
            }
        }
    }

    pub(self) fn unify_in_place(
        &mut self,
        other: &Self,
        env: &mut UnifyEnv,
    ) -> Result<(), (Self, Self)> {
        use Term::*;

        if self == other {
            return Ok(());
        }

        let err = |self_, other_: &Self| {
            debug!("Unification error: {:?} != {:?}", self_, other_);
            Err((self_, other_.clone()))
        };

        match (self, other) {
            (Var(var1), Var(var2)) => {
                // If the variables are the same, they are already unified
                if var1 == var2 {
                    return Ok(());
                }

                // If the variable is already bound, unify the bound term with the other variable
                if let Some(mut term) = env.get_var(*var1).cloned() {
                    term.unify_in_place(other, env)?;
                } else if let Some(term) = env.get_var(*var2).cloned() {
                    // term.unify_in_place(&Var(*var1), env)?;
                    Var(*var1).unify_in_place(&term, env)?;
                } else {
                    // Bind the variable to the other term
                    env.set_var(*var1, Var(*var2));
                }
            }
            (Var(var), term) => {
                if let Some(mut term2) = env.get_var(*var).cloned() {
                    term2.unify_in_place(term, env)?;
                } else {
                    env.set_var(*var, term.clone());
                }
            }
            (term, Var(var)) => {
                if let Some(term2) = env.get_var(*var).cloned() {
                    term.unify_in_place(&term2, env)?;
                } else {
                    env.set_var(*var, term.clone());
                }
            }

            (App(app1), App(app2)) if app1.func == app2.func && app1.args.len() == app2.args.len() => {
                for (arg1, arg2) in app1.args.iter_mut().zip(app2.args.iter()) {
                    arg1.unify_in_place(arg2, env)?;
                }
            }

            (Complement(term1), Complement(term2)) => {
                term1.unify_in_place(term2, env)?;
            }
            (Complement(term), True) => {
                term.unify_in_place(&False, env)?;
            }
            (Complement(term), False) => {
                term.unify_in_place(&True, env)?;
            }
            (True, Complement(term)) => {
                term.unify(&False, env)?;
            }
            (False, Complement(term)) => {
                term.unify(&True, env)?;
            }
            (Complement(term), term2) => {
                if let Ok(term) = term.unify(term2, env) {
                    return err(Complement(Box::new(term)), term2);
                } else {
                    return Ok(());
                }
            }
            (term1, Complement(term)) => {
                if let Ok(term) = term1.unify(term, env) {
                    return err(term1.clone(), &Complement(Box::new(term)));
                } else {
                    return Ok(());
                }
            }

            (Int(n1), Int(n2)) if n1 == n2 => {}
            (True, True) | (False, False) => {}
            (Str(s1), Str(s2)) if s1 == s2 => {}

            (List(terms1), List(terms2)) if terms1.len() == terms2.len() => {
                // for (term1, term2) in terms1.iter().zip(terms2.iter()) {
                //     term1.unify_in_place(term2, bindings)?;
                // }
                for (term1, term2) in terms1.iter_mut().zip(terms2.iter()) {
                    term1.unify_in_place(term2, env)?;
                }
            }

            // (Set(terms1), Set(terms2)) if terms1.len() == terms2.len() => {
            //     // for (term1, term2) in terms1.iter().zip(terms2.iter()) {
            //     //     term1.unify_in_place(term2, bindings)?;
            //     // }
            //     // for (term1, term2) in terms1.iter_mut().zip(terms2.iter_mut()) {
            //     //     term1.unify_in_place(term2, env)?;
            //     // }
            //     let mut result = BTreeSet::new();
            //     for (term1, term2) in terms1.iter().zip(terms2.iter()) {
            //         result.insert(term1.unify(term2, env)?);
            //     }

            //     *terms1 = result;
            // }

            // (Map(terms1), Map(terms2)) if terms1.len() == terms2.len() => {

            //     let mut result = BTreeMap::new();
            //     for ((key1, val1), (key2, val2)) in terms1.iter().zip(terms2.iter()) {
            //         result.insert(key1.unify(key2, env)?, val1.unify(val2, env)?);
            //     }

            //     *terms1 = result;
            // }

            (Nil, Nil) | (Cut, Cut) => {}

            (self_, Union(other)) if other.len() == 1 => {
                self_.unify_in_place(&other[0], env)?;
            }

            (self_, Intersect(other)) if other.len() == 1 => {
                self_.unify_in_place(&other[0], env)?;
            }

            (Union(terms1), other) => {
                for term1 in terms1.iter() {
                    if let Ok(term) = term1.unify(other, env) {
                        *terms1 = vec![term];
                        return Ok(())
                    }
                }
            }

            (other, Union(terms2)) => {
                for term2 in terms2.iter() {
                    if let Ok(term) = other.unify(term2, env) {
                        *other = Union(vec![term]);
                        return Ok(())
                    }
                }
            }

            (Intersect(terms1), other) => {
                let mut tmp = env.clone();
                for term1 in terms1.iter() {
                    if let Err(_) = term1.unify(other, &mut tmp) {
                        return err(term1.clone(), other);
                    }
                }
                *env = tmp;
                *terms1 = vec![other.clone()];
            }

            (self_, Intersect(terms2)) => {
                let mut tmp = env.clone();
                for term2 in terms2.iter() {
                    if let Err(_) = term2.unify(self_, &mut tmp) {
                        return err(self_.clone(), other);
                    }
                }
                *self_ = Intersect(vec![terms2[0].clone()]);
                *env = tmp;
            }

            // (Intersect(terms1), Intersect(terms2)) => {
            //     let mut result = Vec::new();
            //     for term1 in terms1.iter() {
            //         for term2 in terms2.iter() {
            //             if let Ok(term) = term1.unify(term2, env) {
            //                 result.push(term);
            //             }
            //         }
            //     }
            //     *terms1 = result;
            // }

            (Cons(head1, tail1), Cons(head2, tail2)) => {
                // head1.unify_in_place(head2, bindings)?;
                // tail1.unify_in_place(tail2, bindings)?;

                head1.unify_in_place(head2, env)?;
                tail1.unify_in_place(tail2, env)?;
            }
            
            (Cons(head, tail), List(list)) => {
                /*
                if list.is_empty() {
                    // Unify the head and tail with nil
                    head.unify_in_place(&Term::Nil, bindings)?;
                    tail.unify_in_place(&Term::Nil, bindings)?;
                    return Ok(());
                }

                if list.len() == 2 {
                    // Unify the head and tail with a single element list
                    head.unify_in_place(&list[0], bindings)?;
                    tail.unify_in_place(&list[1], bindings)?;
                    return Ok(());
                }

                if list.len() == 1 {
                    // Unify the head and tail with a single element list
                    head.unify_in_place(&list[0], bindings)?;
                    tail.unify_in_place(&Term::Nil, bindings)?;
                    return Ok(());
                }

                head.unify_in_place(&list[0], bindings)?;
                tail.unify_in_place(&Term::List(list[1..].to_vec()), bindings)?;
                 */

                if list.is_empty() {
                    // Unify the head and tail with nil
                    head.unify_in_place(&Term::Nil, env)?;
                    tail.unify_in_place(&Term::Nil, env)?;
                    return Ok(());
                }

                if list.len() == 2 {
                    // Unify the head and tail with a single element list
                    head.unify_in_place(&list[0], env)?;
                    tail.unify_in_place(&list[1], env)?;
                    return Ok(());
                }

                if list.len() == 1 {
                    // Unify the head and tail with a single element list
                    head.unify_in_place(&list[0], env)?;
                    tail.unify_in_place(&Term::Nil, env)?;
                    return Ok(());
                }

                head.unify_in_place(&list[0], env)?;
                tail.unify_in_place(&Term::List(list[1..].to_vec()), env)?;
            }

            (List(list), Cons(head, tail)) => {
                if list.is_empty() {
                    head.unify(&Term::Nil, env)?;
                    tail.unify(&Term::Nil, env)?;
                }

                if list.len() == 2 {
                    list[0].unify_in_place(head, env)?;
                    list[1].unify_in_place(tail, env)?;
                }

                if list.len() == 1 {
                    list[0].unify_in_place(head, env)?;
                    tail.unify(&Term::Nil, env)?;
                }

                list[0].unify_in_place(head, env)?;
                Term::List(list[1..].to_vec()).unify_in_place(tail, env)?;
            }

            (self_, _) => {
                debug!("Type mismatch: {:?} != {:?}", self_, other);
                return err(self_.clone(), other);
            }
        }
        Ok(())
    }

    fn has_free_vars(&self) -> bool {
        use Term::*;

        match self {
            Sym(_) => false,
            Var(_) => true,
            Complement(term) => term.has_free_vars(),
            Equal(lhs, rhs) | NotEqual(lhs, rhs) => lhs.has_free_vars() || rhs.has_free_vars(),
            App(app) => app.args.iter().any(|arg| arg.has_free_vars()),
            Int(_) | Str(_) | Nil | Cut | True | False => false,
            List(terms) => terms.iter().any(|term| term.has_free_vars()),
            Cons(head, tail) => head.has_free_vars() || tail.has_free_vars(),
            // Set(terms) => terms.iter().any(|term| term.has_free_vars()),
            // Map(terms) => terms
            //     .iter()
            //     .any(|(key, val)| key.has_free_vars() || val.has_free_vars()),
            Union(terms) => terms.iter().any(|term| term.has_free_vars()),
            Intersect(terms) => terms.iter().any(|term| term.has_free_vars()),
        }
    }

    fn free_vars(&self, vars: &mut BTreeSet<Var>) {
        use Term::*;

        match self {
            Sym(_) => {}
            Var(var) => {
                vars.insert(*var);
            }
            App(app) => {
                for arg in app.args.iter() {
                    arg.free_vars(vars);
                }
            }
            List(terms) => {
                for term in terms.iter() {
                    term.free_vars(vars);
                }
            }
            Cons(head, tail) => {
                head.free_vars(vars);
                tail.free_vars(vars);
            }
            // Set(terms) => {
            //     for term in terms.iter() {
            //         term.free_vars(vars);
            //     }
            // }
            // Map(terms) => {
            //     for (key, val) in terms.iter() {
            //         key.free_vars(vars);
            //         val.free_vars(vars);
            //     }
            // }
            Union(terms) => {
                for term in terms.iter() {
                    term.free_vars(vars);
                }
            }
            Intersect(terms) => {
                for term in terms.iter() {
                    term.free_vars(vars);
                }
            }
            Equal(lhs, rhs) | NotEqual(lhs, rhs) => {
                lhs.free_vars(vars);
                rhs.free_vars(vars);
            }
            Complement(term) => {
                term.free_vars(vars);
            }
            Int(_) | Str(_) | Nil | Cut | True | False => {}
        }
    }

    fn free_params(&self, params: &mut BTreeSet<Var>) {
        use Term::*;

        match self {
            Sym(_) => {}
            Var(_) => {}
            Complement(term) => term.free_params(params),
            App(app) => app.params(params),
            List(terms) => {
                for term in terms.iter() {
                    term.free_params(params);
                }
            }
            Cons(head, tail) => {
                head.free_params(params);
                tail.free_params(params);
            }
            // Set(terms) => {
            //     for term in terms.iter() {
            //         term.free_params(params);
            //     }
            // }
            // Map(terms) => {
            //     for (key, val) in terms.iter() {
            //         key.free_params(params);
            //         val.free_params(params);
            //     }
            // }
            Union(terms) => {
                for term in terms.iter() {
                    term.free_params(params);
                }
            }
            Intersect(terms) => {
                for term in terms.iter() {
                    term.free_params(params);
                }
            }
            Equal(lhs, rhs) | NotEqual(lhs, rhs) => {
                lhs.free_params(params);
                rhs.free_params(params);
            }
            Int(_) | Str(_) | Nil | Cut | True | False => {}
        }
    }

    pub fn reduce(&self, env: &UnifyEnv) -> Self {
        let mut term = self.clone();
        term.reduce_in_place(env);
        term
    }

    pub fn reduce_in_place(&mut self, env: &UnifyEnv) {
        self.substitute(&env.bindings);

        match self {
            Self::Equal(ref a, ref b) => {
                let a = a.clone();
                let b = b.clone();
                *self = if a == b {
                    Self::True
                } else {
                    Self::False
                }
            },
            Self::NotEqual(ref a, ref b) => {
                let a = a.clone();
                let b = b.clone();
                *self = if a != b {
                    Self::True
                } else {
                    Self::False
                }
            },
            Self::Complement(term) => {
                term.reduce_in_place(env);
                *self = match term.as_ref() {
                    Self::True => Self::False,
                    Self::False => Self::True,
                    other => Self::Complement(Box::new(other.clone())),
                }
            },
            // If all elements are equal
            Self::Intersect(terms) if terms.iter().all(|term| term == &terms[0]) => {
                *self = terms[0].clone();
                self.reduce_in_place(env);
            }
            Self::Union(terms) if terms.iter().all(|term| term == &terms[0]) => {
                *self = terms[0].clone();
                self.reduce_in_place(env);
            }

            Self::Intersect(terms) if terms.len() == 1 => {
                *self = terms[0].clone();
                self.reduce_in_place(env);
            }
            Self::Union(terms) if terms.len() == 1 => {
                *self = terms[0].clone();
                self.reduce_in_place(env);
            }
            _ => {}
        };

        for child in self.subterms_mut() {
            child.reduce_in_place(env);
        }
    }

    pub fn subterms(&self) -> Vec<&Self> {
        use Term::*;

        let mut subterms = vec![];
        match self {
            Complement(term) => {
                subterms.extend(term.subterms());
            }
            Sym(_) | Var(_) | Int(_) | Str(_) | Nil | Cut | True | False => {}
            App(app) => {
                subterms.extend(app.args.iter().flat_map(|arg| arg.subterms()));
            }
            List(terms) => {
                subterms.extend(terms.iter().flat_map(|term| term.subterms()));
            }
            Cons(head, tail) => {
                subterms.extend(head.subterms());
                subterms.extend(tail.subterms());
            }
            // Set(terms) => {
            //     subterms.extend(terms.iter().flat_map(|term| term.subterms()));
            // }
            // Map(terms) => {
            //     subterms.extend(terms.iter().flat_map(|(key, val)| key.subterms().chain(val.subterms())));
            // }
            Union(terms) | Intersect(terms) => {
                subterms.extend(terms.iter().flat_map(|term| term.subterms()));
            }
            Equal(lhs, rhs) | NotEqual(lhs, rhs) => {
                subterms.extend(lhs.subterms());
                subterms.extend(rhs.subterms());
            }
        }

        subterms
    }

    pub fn subterms_mut(&mut self) -> Vec<&mut Self> {
        use Term::*;

        let mut subterms: Vec<&mut Term> = vec![];
        match self {
            Complement(term) => {
                subterms.extend(term.subterms_mut());
            }
            Sym(_) | Var(_) | Int(_) | Str(_) | Nil | Cut | True | False => {}
            App(app) => {
                subterms.extend(app.args.iter_mut().flat_map(|arg| arg.subterms_mut()));
            }
            List(terms) => {
                subterms.extend(terms.iter_mut().flat_map(|term| term.subterms_mut()));
            }
            Cons(head, tail) => {
                subterms.extend(head.subterms_mut());
                subterms.extend(tail.subterms_mut());
            }
            // Map(terms) => {
            //     subterms.extend(terms.iter_mut().flat_map(|(key, val)| key.subterms_mut().chain(val.subterms_mut())));
            // }
            Union(terms) | Intersect(terms) => {
                subterms.extend(terms.iter_mut().flat_map(|term| term.subterms_mut()));
            }
            Equal(lhs, rhs) | NotEqual(lhs, rhs) => {
                subterms.extend(lhs.subterms_mut());
                subterms.extend(rhs.subterms_mut());
            }
        }

        subterms
    }

    pub fn substitute(&mut self, bindings: &BTreeMap<Var, Term>) {
        use Term::*;

        if bindings.is_empty() || !self.has_free_vars() {
            return;
        }

        match self {
            Complement(term) => {
                term.substitute(bindings);
            }
            Equal(lhs, rhs) | NotEqual(lhs, rhs) => {
                lhs.substitute(bindings);
                rhs.substitute(bindings);
            }
            Var(var) => {
                if let Some(term) = bindings.get(var) {
                    *self = term.clone();
                }
            }
            App(app) => {
                for arg in app.args.iter_mut() {
                    arg.substitute(bindings);
                }
            }
            List(terms) => {
                for term in terms.iter_mut() {
                    term.substitute(bindings);
                }
            }
            Cons(head, tail) => {
                head.substitute(bindings);
                tail.substitute(bindings);
            }
            // Map(terms) => {
            //     *terms = terms
            //         .iter()
            //         .map(|(key, val)| {
            //             let mut key = key.clone();
            //             let mut val = val.clone();
            //             key.substitute(bindings);
            //             val.substitute(bindings);
            //             (key, val)
            //         })
            //         .collect();
            // }

            Union(terms) => {
                for term in terms.iter_mut() {
                    term.substitute(bindings);
                }
            }

            Intersect(terms) => {
                for term in terms.iter_mut() {
                    term.substitute(bindings);
                }
            }

            Cut | Nil | Int(_) | Str(_) | Sym(_) | True | False => {}
        }
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

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AppTerm {
    pub func: Symbol,
    pub args: Vec<Term>,
}

impl AppTerm {
    pub fn new(func: impl ToString, args: Vec<Term>) -> Self {
        let func = Symbol::new(func.to_string().as_str());
        AppTerm { func, args }
    }

    fn params(&self, params: &mut BTreeSet<Var>) {
        for arg in self.args.iter() {
            arg.free_vars(params);
        }
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

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Rule {
    /// The head of the rule
    /// This is the term that is being defined
    pub head: Term,

    /// The terms that must be true for the head to be true
    pub tail: Vec<Term>,
}

impl Rule {
    pub fn fact(head: impl ToString, args: Vec<Term>) -> Self {
        Rule {
            head: AppTerm::new(head, args).into(),
            tail: vec![],
        }
    }

    pub fn might_apply_to(&self, term: &Term) -> bool {
        match (&self.head, term) {
            (Term::App(app1), Term::App(app2)) => {
                app1.func == app2.func && app1.args.len() == app2.args.len()
            }
            _ => true,
        }
    }

    pub fn is_recursive(&self) -> bool {
        match self.head {
            Term::App(ref app) => {
                app.args.iter().any(|arg| arg.has_application_of(app))
                || (!self.tail.is_empty() && self.tail.iter().any(|term| term.has_application_of(app)))
            },
            _ => false,
        }
    }

    pub fn is_invalid(&self) -> bool {
        match self.head {
            Term::App(ref app) => {
                // All the arguments must be variables
                app.args.iter().all(|arg| match arg {
                    Term::Var(_) => true,
                    _ => false,
                })  && !self.tail.is_empty()
                    && self.tail.iter().all(|term| term.has_application_of(app))
            },
            _ => false,
        }
    }

    pub fn size(&self) -> usize {
        self.head.size() + self.tail.iter().map(|term| term.size()).sum::<usize>()
    }

    /// Constrain a rule by adding a term to the tail
    pub fn when(mut self, head: impl ToString, args: Vec<Term>) -> Self {
        self.tail.push(app!(head, args));
        self
    }
    /*
    pub fn refresh(&mut self, bindings: &BTreeMap<Var, Term>) {
        // Get all the params in the rule
        let mut params = BTreeSet::new();
        self.head.free_params(&mut params);
        // Now, replace all the params with fresh vars
        let mut new_bindings = BTreeMap::new();
        for param in params.iter() {
            // If the param is not in the bindings, then we don't need to refresh it
            if bindings.contains_key(param) {
                continue;
            }

            let fresh = param.refresh();
            new_bindings.insert(*param, Term::from(fresh));
        }
        self.head.substitute(&new_bindings);
        for term in self.tail.iter_mut() {
            term.substitute(&new_bindings);
        }
    }
     */


    // /// A simple helper to refresh (rename) the variables in a Rule so that
    // /// each usage has distinct variable IDs. If your `Symbol` system already
    // /// gives fresh IDs for newly created variables, you can simply re-create them.
    // fn refresh_rule_vars(rule: &mut Rule) {
    //     // Gather all variables
    //     let mut vars_in_rule = BTreeSet::new();
    //     rule.head.free_vars(&mut vars_in_rule);
    //     for t in &rule.tail {
    //         t.free_vars(&mut vars_in_rule);
    //     }

    //     // For each var, make a new variable with a fresh symbol name
    //     let mut rename_map = BTreeMap::new();
    //     for old_var in vars_in_rule {
    //         let new_var = old_var.refresh();
    //         rename_map.insert(old_var, Term::Var(new_var));
    //     }

    //     // Substitute them in the rule
    //     rule.head.substitute(&rename_map);
    //     for t in rule.tail.iter_mut() {
    //         t.substitute(&rename_map);
    //     }
    // }
    pub fn refresh(&mut self) {
        // Gather all variables
        let mut vars_in_rule = BTreeSet::new();
        self.head.free_vars(&mut vars_in_rule);
        for t in &self.tail {
            t.free_vars(&mut vars_in_rule);
        }

        // For each var, make a new variable with a fresh symbol name
        let mut rename_map = BTreeMap::new();
        for old_var in vars_in_rule {
            let new_var = old_var.refresh();
            rename_map.insert(old_var, Term::Var(new_var));
        }

        // Substitute them in the rule
        self.head.substitute(&rename_map);
        for t in self.tail.iter_mut() {
            t.substitute(&rename_map);
        }
    }


    pub fn substitute(&mut self, bindings: &BTreeMap<Var, Term>) {
        self.head.substitute(bindings);
        for term in self.tail.iter_mut() {
            term.substitute(bindings);
        }
    }
}

impl From<Term> for Rule {
    fn from(term: Term) -> Self {
        Rule {
            head: term,
            tail: vec![],
        }
    }
}

impl FromStr for Rule {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Use nom::error::convert_error to get better error messages
        parse_rule(s)
            .map_err(|e| match e {
                nom::Err::Error(e) | nom::Err::Failure(e) => nom::error::convert_error(s, e),
                nom::Err::Incomplete(_) => "Incomplete input".to_string(),
            })
            .map(|(_, rule)| rule)
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Query {
    pub goals: Vec<Term>,
}

impl Query {
    pub fn new(goals: Vec<Term>) -> Self {
        Query { goals }
    }

    pub fn iter(&self) -> impl Iterator<Item = &Term> {
        self.goals.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Term> {
        self.goals.iter_mut()
    }

    pub fn push(&mut self, goal: Term) {
        self.goals.push(goal);
    }

    pub fn pop(&mut self) -> Option<Term> {
        self.goals.pop()
    }

    pub fn free_vars(&self, vars: &mut BTreeSet<Var>) {
        for goal in self.goals.iter() {
            goal.free_vars(vars);
        }
    }
}

impl FromStr for Query {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parse_query(s)
            .map_err(|e| match e {
                nom::Err::Error(e) | nom::Err::Failure(e) => nom::error::convert_error(s, e),
                nom::Err::Incomplete(_) => "Incomplete input".to_string(),
            })
            .map(|(_, query)| query)
    }
}

#[macro_export]
macro_rules! term {
    // Match a list
    ([ $($val:expr),* $(,)? ]) => {
        Term::List(vec![$(term!($val)),*])
    };

    // // Match a set
    // ({ $($val:expr),* $(,)? }) => {
    //     Term::Set({
    //         let mut set = BTreeSet::new();
    //         $(set.insert(term!($val));)*
    //         set
    //     })
    // };

    // // Match a map
    // ({ $($key:expr => $val:expr),* $(,)? }) => {
    //     Term::Map({
    //         let mut map = BTreeMap::new();
    //         $(map.insert(term!($key), term!($val));)*
    //         map
    //     })
    // };

    // Match a union
    (v [ $($val:expr),* $(,)? ]) => {
        Term::Union(vec![$(term!($val)),*])
    };

    // Match an intersection
    (^ [ $($val:expr),* $(,)? ]) => {
        Term::Intersect(vec![$(term!($val)),*])
    };

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
        let symbol = Symbol::from_id(self.id);
        write!(f, "{}", symbol)
    }
}

impl Display for Var {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let symbol = Symbol::from_id(self.id);
        write!(f, "{}", symbol)
    }
}

impl Display for Term {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{:?}", self)
    }
}

impl Debug for Term {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        match self {
            Term::Complement(term) => write!(f, "~{:?}", term),
            Term::Sym(sym) => write!(f, "{}", sym),
            Term::Var(var) => write!(f, "{:?}", var),
            Term::App(app) => write!(f, "{:?}", app),
            Term::Int(n) => write!(f, "{}", n),
            Term::True => write!(f, "true"),
            Term::False => write!(f, "false"),
            Term::Str(s) => write!(f, "{:?}", s),
            Term::List(terms) => {
                write!(f, "[")?;
                for (i, term) in terms.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:?}", term)?;
                }
                write!(f, "]")
            }
            Term::Cons(head, tail) => {
                write!(f, "[{:?}|{:?}]", head, tail)
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
            Term::Union(terms) => {
                write!(f, "v [")?;
                for term in terms.iter() {
                    write!(f, "{:?} ", term)?;
                }
                write!(f, "]")
            }
            Term::Intersect(terms) => {
                write!(f, "^ [")?;
                for term in terms.iter() {
                    write!(f, "{:?} ", term)?;
                }
                write!(f, "]")
            }
            Term::Nil => write!(f, "nil"),
            Term::Cut => write!(f, "#"),
            Term::Equal(lhs, rhs) => write!(f, "{:?} = {:?}", lhs, rhs),
            Term::NotEqual(lhs, rhs) => write!(f, "{:?} != {:?}", lhs, rhs),
        }
    }
}

impl Debug for AppTerm {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{}(", self.func)?;
        for (i, arg) in self.args.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:?}", arg)?;
        }
        write!(f, ")")
    }
}

impl Debug for Rule {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{:?}", self.head)?;
        if !self.tail.is_empty() {
            write!(f, " :- ")?;
            for (i, term) in self.tail.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:?}", term)?;
            }
        }
        write!(f, ".")
    }
}

impl Debug for Query {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "?- ")?;
        for (i, goal) in self.goals.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:?}", goal)?;
        }
        write!(f, ".")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        //     let mut map = BTreeMap::new();
        //     map.insert(Term::Str("foo".to_string()), Term::Int(42));
        //     map.insert(Term::Str("bar".to_string()), Term::True);
        //     map.insert(Term::Str("baz".to_string()), Term::Str("qux".to_string()));
        //     map.insert(
        //         var("test"),
        //         Term::Set({
        //             let mut set = BTreeSet::new();
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

        let mut env = UnifyEnv::default();
        let list = term!([1, 2]);

        cons.unify_in_place(&list, &mut env).unwrap();

        assert_eq!(cons, term!(cons 1; 2));


        let mut cons = term!(
            cons var("X");
            term!(cons var("Y"); var("Z")));
        let list = term!([1, 2, 3]);

        cons.unify_in_place(&list, &mut env).unwrap();

        assert_eq!(cons, term!(cons 1; term!(cons 2; 3)));

        let mut cons = term!(
            cons var("X");
            term!(cons var("Y"); var("X")));
        let list = term!([1, 2, 1]);

        cons.unify_in_place(&list, &mut env).unwrap();

        assert_eq!(cons, term!(cons 1; term!(cons 2; 1)));
    }

    /// Test unification
    #[test]
    fn test_unification() {
        let mut term1 = app!("foo", [var("X"), var("Y")]);
        let term2 = app!("foo", [42, true]);
        let mut env = UnifyEnv::default();

        let mut ground_truth = BTreeMap::new();
        ground_truth.insert("X".into(), 42.into());
        ground_truth.insert("Y".into(), true.into());
        term1.unify_in_place(&term2, &mut env).unwrap();
        assert_eq!(env.bindings, ground_truth);

        let mut term1 = app!("foo", [var("X"), var("Y")]);
        let term2 = app!("foo", [42, var("Z")]);
        term1.unify_in_place(&term2, &mut env).unwrap();

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

        term1.unify_in_place(&term2, &mut env).unwrap();

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

        term1.unify_in_place(&term2, &mut env).unwrap();

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
        let mut env = UnifyEnv::default();
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
        term1.unify_in_place(&term2, &mut env).unwrap();

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
        let mut env = UnifyEnv::default();
        
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
        assert!(term1.unify_in_place(&term2, &mut env).is_err());
    }

    /// Test successful unification of intersection expressions requiring all operands to match.
    #[test]
    fn test_intersection_unification_success() {
        let mut env = UnifyEnv::default();

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
        term1.unify_in_place(&term2, &mut env).unwrap();

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
        let mut env = UnifyEnv::default();

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
        assert!(term1.unify_in_place(&term2, &mut env).is_err());
    }
}
