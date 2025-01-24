//! solver.rs
// pub mod naive;
pub mod dfs;

use crate::{Query, Solution, Term, UnifyEnv, Var};
use std::collections::{BTreeMap, BTreeSet, HashSet};

pub trait Solver {
    /// Unify two terms
    /// Returns an error if the terms cannot be unified
    fn unify(&mut self, term1: &Term, term2: &Term) -> Result<Term, (Term, Term)> {
        let mut env = UnifyEnv::default();
        term1.unify(term2, &mut env)
    }

    /// Solve a query and return a list of variable bindings
    fn solve_vars(&mut self, query: &Query, max_solutions: usize) -> Result<BTreeSet<Solution>, String>;

    /// Solve a query and return the full goal terms with the variable bindings substituted
    fn solve(&mut self, mut query: Query, max_solutions: usize) -> Result<Vec<Term>, String> {
        let bindings = self.solve_vars(&query, max_solutions)?;
        for goal in query.goals.iter_mut() {
            for env in bindings.iter() {
                goal.reduce(&env.clone().into());
            }
        }
        Ok(query.goals)
    }

    fn can_prove(&mut self, query: &Query) -> bool {
        self.solve_vars(query, 1).is_ok()
    }
}