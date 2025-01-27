use super::*;
use std::collections::{HashMap, HashSet};

use std::fmt::{Debug, Display, Formatter, Result as FmtResult};
use std::hash::Hash;
use std::str::FromStr;
use tracing::debug;

/// A rule in the logic program.
/// 
/// A rule is a head term, followed by a list of tail terms.
/// 
/// The rule is true if the head is true, and all the tail terms are true.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Rule {
    /// The head of the rule
    /// This is the term that is being defined
    pub head: Term,

    /// The terms that must be true for the head to be true
    pub tail: Vec<Term>,
}

impl Rule {
    /// Define a fact, which is a rule with no conditions.
    /// 
    /// This rule is treated as always true by the solver.
    pub fn fact(head: impl ToString, args: Vec<Term>) -> Self {
        Rule {
            head: AppTerm::new(head, args).into(),
            tail: vec![],
        }
    }


    /// Apply a rule to a term in an environment.
    /// 
    /// The `original_term` is the term that the rule is being applied to.
    /// 
    /// The query is passed to allow the rule to insert the new goals, which
    /// then must be proven by the solver for the rule to be considered true.
    /// 
    /// The environment is passed to allow the rule to perform unification,
    /// and to allow the rule to prove complements false.
    pub fn apply<S>(&self, original_term: &Term, query: &mut Query, env: &mut Env<S>) -> bool where S: Solver {
        if !self.might_apply_to(original_term) {
            return false;
        }

        if self.head == *original_term && self.tail.is_empty() {
            // If the head is the same as the term, then the rule is trivially true
            // Remove the term from the query
            query.remove_goal(&original_term);

            return true;
        }


        let mut negated = false;
        let mut term = original_term.clone();
        if let Term::Complement(inner) = term {
            // error!("Bailing out of negative term {}", original_term);
            // return false;
            term = *inner;
            negated = true;
        }

        // query.reduce_in_place(&env);

        let mut new_rule = self.clone();
        // let mut used_vars = HashSet::new();
        // env.used_vars(&mut used_vars);
        // new_rule.refresh_variables(used_vars);
        new_rule.refresh();

        // Try to unify the head of the rule with the term
        let mut new_term = term.clone();
        let mut new_env = env.clone();
        let old_query = query.clone();
        if let Ok(_) = new_term.unify_in_place(&new_rule.head, &mut new_env, false) {
            if negated && self.tail.is_empty() {
                // Then the negation fails, because the head is true.
                // Add false to the query's goals.
                debug!("{original_term} is absurd by rule {self}.");
                query.remove_goal(&original_term);
                query.add_positive_goal(Term::False);
                query.reduce_in_place(&new_env);
                *env = new_env;
                return true;
            }

            // debug!("Unifying {} with {}", new_term, new_rule.head);
            query.remove_goal(&original_term);

            // If the head unifies, then add the tail to the query
            let mut can_prove_one_false = false;
            for term in new_rule.tail.iter() {
                if negated {
                    debug!("Negating term {}", term);
                    // Try to prove one subgoal false
                    let subquery = Query::new(vec![term.clone()]);
                    if new_env.prove_false(&subquery).is_ok() {
                        debug!("Proved {} false", term);
                        can_prove_one_false = true;
                    } else {
                        debug!("Failed to prove {} false", term);
                        // info!("Env is {:#?}", new_env);
                        query.add_positive_goal(term.clone());
                    }
                } else {
                    // If we should negate, only *one* of the terms in the tail has to be false
                    query.add_positive_goal(term.clone());
                }
            }

            if negated && !can_prove_one_false {
                // info!("Negation failed");
                debug!("{original_term} is absurd, could not prove antecedents.");
                query.add_positive_goal(Term::False);
                query.reduce_in_place(&new_env);
                *env = new_env;
                return true;
            } else if negated {
                debug!("{original_term} is true by rule {self}.");
            }

            // query.reduce_in_place(&new_env);
            
            if *query == old_query && new_env == *env {
                // warn!("Rule {} did not change the query", self);
                // warn!("Query: {}", query);
                // warn!("Env: {:?}", env);
                return false;
            }
            *env = new_env;
            true
        } else {
            debug!("Failed to unify {} with {}", new_term, new_rule.head);
            false
        }
    }

    /// Might this rule apply to a term?
    /// 
    /// If the term is an application, and the function name and arity match, then
    /// the rule might apply.
    /// If the function name and arity do not match, then the rule cannot apply.
    /// 
    /// If the term is not an application, then the rule might apply.
    /// The term might be a variable, for example, which could be unified with the head of the rule.
    pub fn might_apply_to(&self, term: &Term) -> bool {
        match (&self.head, term) {
            (Term::App(app1), Term::App(app2)) => {
                app1.func == app2.func && app1.args.len() == app2.args.len()
            }
            _ => true,
        }
    }

    /// Is this rule recursive?
    /// A rule is recursive if any of the following are true:
    /// 1. The arguments of the head contain an application of the head function
    /// 2. The tail contains an application of the head function
    pub fn is_recursive(&self) -> bool {
        match self.head {
            Term::App(ref app) => {
                app.args.iter().any(|arg| arg.has_application_of(app))
                || (!self.tail.is_empty() && self.tail.iter().any(|term| term.has_application_of(app)))
            },
            _ => false,
        }
    }

    /// Is this rule invalid?
    /// 
    /// A rule is invalid if all the arguments of the head are variables (no progress can be made by the head matching),
    /// and all the terms in the tail contain an application of the head function.
    /// 
    /// This is invalid because the rule will never be able to make progress, and will always be stuck in a loop.
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

    /// Get the size of the rule.
    /// 
    /// The size of a rule is the sum of the sizes of the head and the tail terms.
    pub fn size(&self) -> usize {
        self.head.size() + self.tail.iter().map(|term| term.size()).sum::<usize>()
    }

    /// Constrain a rule by adding a term to the tail.
    /// 
    /// This makes the head of the rule *only true* if this condition is also true.
    pub fn when(mut self, head: impl ToString, args: Vec<Term>) -> Self {
        self.tail.push(AppTerm::new(head, args).into());
        self
    }
    /*
    pub fn refresh(&mut self, bindings: &HashMap<Var, Term>) {
        // Get all the params in the rule
        let mut params = HashSet::new();
        self.head.used_params(&mut params);
        // Now, replace all the params with fresh vars
        let mut new_bindings = HashMap::new();
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

     /// Refresh all the variables in the rule.
     /// 
     /// This is useful when the rule is being applied to a term, and we don't want to accidentally
     /// have name collisions with the variables in the term.
    pub fn refresh(&mut self) {
        // Gather all variables
        let mut vars_in_rule = HashSet::new();
        self.head.used_vars(&mut vars_in_rule);
        for t in &self.tail {
            t.used_vars(&mut vars_in_rule);
        }

        // For each var, make a new variable with a fresh symbol name
        let mut rename_map = HashMap::new();
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

    /// Refresh only the variables in the rule that are in the set of variables to refresh.
    /// 
    /// This is useful when we want to avoid refreshing variables unnecessarily.
    pub fn refresh_variables(&mut self, variables_to_refresh: HashSet<Var>) {
        // Gather all variables
        let mut vars_in_rule = HashSet::new();
        self.head.used_vars(&mut vars_in_rule);
        for t in &self.tail {
            t.used_vars(&mut vars_in_rule);
        }

        // For each var, make a new variable with a fresh symbol name
        let mut rename_map = HashMap::new();
        for old_var in vars_in_rule {
            if variables_to_refresh.contains(&old_var) {
                let new_var = old_var.refresh();
                rename_map.insert(old_var, Term::Var(new_var));
            }
        }

        // Substitute them in the rule
        self.head.substitute(&rename_map);
        for t in self.tail.iter_mut() {
            t.substitute(&rename_map);
        }
    }

    /// Substitute all the variables in the rule with the given bindings.
    pub fn substitute(&mut self, bindings: &HashMap<Var, Term>) {
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
impl Display for Rule {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{}", self.head)?;
        if !self.tail.is_empty() {
            write!(f, " :- ")?;
            for (i, term) in self.tail.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", term)?;
            }
        }
        write!(f, ".")
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use crate::solvers::*;

    #[test]
    fn test_application() {
        let rule: Rule = "f(X) :- g(X), h(X).".parse().unwrap();

        let mut query: Query = "?- f(1).".parse().unwrap();

        let mut env = Env::<DefaultSolver>::new(&[rule.clone()]);
        println!("{}", query);

        assert!(rule.apply(&query[0].clone(), &mut query, &mut env));

        println!("{}", query);
        println!("{:#?}", env);
    }

    #[test]
    fn test_application2() {
        let rules: Vec<Rule> = vec![
            "is_nat(0).".parse().unwrap(),
            "is_nat(s(X)) :- is_nat(X).".parse().unwrap(),
        ];

        let query: Query = "?- ~is_nat(s(s(s(0)))).".parse().unwrap();

        let env = Env::<DefaultSolver>::new(&rules);

        // env.apply_rules_to_query(&mut query);
        // env.apply_rules_to_query(&mut query);
        // env.apply_rules_to_query(&mut query);
        // let rule = rules[3].clone();
        // rule.apply(&query[0].clone(), &mut query, &mut env);

        println!("{}", query);
        println!("{:#?}", env);
    }

    #[test]
    fn test_application3() {
        let rules: Vec<Rule> = vec![
            "is_nat(0).".parse().unwrap(),
            "is_nat(s(X)) :- is_nat(X).".parse().unwrap(),
        ];

        let mut query: Query = "?- is_nat(s(s(s(s(0))))).".parse().unwrap();

        let mut env = Env::<DefaultSolver>::new(&rules);
        assert!(env.apply_rule_to_query(1, &mut query));
        println!("{}", query);
        assert!(env.apply_rule_to_query(1, &mut query));
        println!("{}", query);
        assert!(env.apply_rule_to_query(1, &mut query));
        println!("{}", query);
        assert!(env.apply_rule_to_query(1, &mut query));
        println!("{}", query);
        assert!(env.apply_rule_to_query(0, &mut query));
        println!("{}", query);
    }

    #[test]
    fn test_nested_application() {
        let rule: Rule = "f(g(X)) :- h(X).".parse().unwrap();

        let mut query: Query = "?- f(X).".parse().unwrap();

        let mut env = Env::<DefaultSolver>::new(&[rule.clone()]);
        println!("{:?}", query);

        assert!(rule.apply(&query[0].clone(), &mut query, &mut env));

        println!("{:?}", query);
        println!("{:#?}", env);
    }

    #[test]
    fn test_nested_application2() {
        let rules: Vec<Rule> = vec![
            "is_nat(0).".parse().unwrap(),
            "is_nat(s(X)) :- is_nat(X).".parse().unwrap(),
            "add(X, 0, X) :- is_nat(X).".parse().unwrap(),
            "add(X, s(Y), s(Z)) :- add(X, Y, Z).".parse().unwrap(),
            "mul(X, 0, 0) :- is_nat(X).".parse().unwrap(),
            "mul(X, s(Y), Z) :- mul(X, Y, W), add(X, W, Z).".parse().unwrap(),
        ];

        let mut query: Query = "?- add(A, B, s(s(s(s(0))))).".parse().unwrap();

        let mut env = Env::<DefaultSolver>::new(&rules);

        env.apply_rule_to_query(3, &mut query);
        // let rule = rules[3].clone();
        // rule.apply(&query[0].clone(), &mut query, &mut env);

        println!("{}", query);
        println!("{:#?}", env);
    }
}