//! solver.rs
// pub mod naive;
// pub mod dfs;

use crate::*;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

/// A solver that saves and reuses rule applications and solutions,
/// according to whatever strategy it implements.
pub trait Solver: Default + Clone + Debug + Eq {
    /// Reset the solver's state, clearing any saved rule applications or solutions
    fn reset(&mut self) {
        // Do nothing by default
    }

    /// Save a rule application for later reuse
    fn save_rule_application(&mut self, _env: Env<Self>, _before_query: &Query, _new_env: Env<Self>, _new_query: &Query) {
        // Do nothing by default
    }

    /// Attempt to use a saved rule application, returning true if one was found
    fn use_saved_rule_application(&self, _env: &mut Env<Self>, _query: &mut Query) -> bool {
        false
    }

    /// Save a set of solutions for later reuse
    fn save_solutions(&mut self, _env: Env<Self>, _query: Query, _solution: HashSet<Solution>) {
        // Do nothing by default
    }

    /// Attempt to use saved solutions, returning them if found
    fn use_saved_solutions(&self, _env: &Env<Self>, _query: &Query) -> Option<HashSet<Solution>> {
        None
    }
}

/// A solver that performs no memoization
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct DefaultSolver;
impl Solver for DefaultSolver {}


/// A solver that memoizes rule solutions
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct MemoizingSolver {
    pub memoized_solutions: HashMap<(Env<Self>, Query), HashSet<Solution>>,
}

impl Solver for MemoizingSolver {
    fn reset(&mut self) {
        self.memoized_solutions.clear();
    }

    fn save_solutions(&mut self, env: Env<Self>, query: Query, solutions: HashSet<Solution>) {
        self.memoized_solutions.insert((env, query.clone()), solutions);
    }

    fn use_saved_solutions(&self, env: &Env<Self>, query: &Query) -> Option<HashSet<Solution>> {
        let (env, query) = (env.clone(), query.clone());
        self.memoized_solutions.get(&(env, query)).cloned()
    }
}