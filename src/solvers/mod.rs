//! solver.rs
// pub mod naive;
// pub mod dfs;

use crate::*;
use std::collections::{BTreeMap, HashSet};
use std::fmt::{Debug, Formatter, Result as FmtResult};


pub trait Solver: Default + Clone + Debug + Eq {
    // fn prove_goal_true(&self, env: &mut Env<Self>, goal: &Term) -> Result<Solution, HashSet<Var>>;
    // fn prove_goal_false(&self, env: &mut Env<Self>, goal: &Term) -> Result<Solution, HashSet<Var>>;

    // fn can_prove_goal_true(&self, env: &Env<Self>, goal: &Term) -> bool {
    //     let mut env = env.clone();
    //     let goal = goal.reduce(&mut env);
    //     self.prove_goal_true(&mut env, &goal).is_ok()
    // }

    // fn can_prove_goal_false(&self, env: &Env<Self>, goal: &Term) -> bool {
    //     let mut env = env.clone();
    //     let goal = goal.reduce(&mut env);
    //     self.prove_goal_false(&mut env, &goal).is_ok()
    // }

    // fn memoize_subgoal(&self, env: &Env<Self>, goal: &Term, result: Result<Solution, HashSet<Var>>) {
    //     // Do nothing by default
    // }

    fn reset(&mut self) {
        // Do nothing by default
    }

    fn save_rule_application(&mut self, env: Env<Self>, before_query: &Query, new_env: Env<Self>, new_query: &Query) {
        // Do nothing by default
    }

    fn use_saved_rule_application(&self, env: &mut Env<Self>, query: &mut Query) -> bool {
        false
    }

    fn save_solutions(&mut self, env: Env<Self>, query: Query, solution: HashSet<Solution>) {
        // Do nothing by default
    }

    fn use_saved_solutions(&self, env: &Env<Self>, query: &Query) -> Option<HashSet<Solution>> {
        None
    }
}

#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct DefaultSolver;
impl Solver for DefaultSolver {}

#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct QuickMemoizingSolver {
    pub memoized_solutions: HashMap<Query, HashSet<Solution>>,
}

impl Solver for QuickMemoizingSolver {
    fn reset(&mut self) {
        self.memoized_solutions.clear();
    }

    fn save_solutions(&mut self, env: Env<Self>, query: Query, solutions: HashSet<Solution>) {
        self.memoized_solutions.insert(query, solutions);
    }

    fn use_saved_solutions(&self, env: &Env<Self>, query: &Query) -> Option<HashSet<Solution>> {
        self.memoized_solutions.get(query).cloned()
    }
}

#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct MemoizingSolver {
    pub memoized_applications: HashMap<(Env<Self>, Query), (Env<Self>, Query)>,
    pub memoized_solutions: HashMap<(Env<Self>, Query), HashSet<Solution>>,
}

impl MemoizingSolver {
    pub fn canonize_application_key(&self, env: &mut Env<Self>, query: &mut Query) {
        // Substitute all variables in the query with their values in the environment
        // env.prune_redundant_variables();
        // query.reduce_in_place(env);
    }

    pub fn canonize_solution_key(&self, env: &mut Env<Self>, query: &mut Query) {
        // Substitute all variables in the solution with their values in the environment
        // let new_vars = env.prune_redundant_variables();
        // query.substitute(&new_vars);
        // query.reduce_in_place(env);
    }
}

impl Solver for MemoizingSolver {
    fn reset(&mut self) {
        self.memoized_applications.clear();
        self.memoized_solutions.clear();
    }

    fn save_rule_application(&mut self, env: Env<Self>, before_query: &Query, new_env: Env<Self>, new_query: &Query) {

        // let mut before_query = before_query.clone();
        // self.canonize_application(&env, &mut before_query);
        let mut env = env.clone();
        let mut before_query = before_query.clone();
        let new_env = new_env.clone();
        let new_query = new_query.clone();

        self.canonize_application_key(&mut env, &mut before_query);

        let memo_key = (env, before_query);
        let memo_value = (new_env, new_query);


        // info!("Memoizing rule application: ({:?}, {}) => ({:?}, {})", memo_key.0, memo_key.1, memo_value.0, memo_value.1);
        self.memoized_applications.insert(memo_key, memo_value);
    }

    fn use_saved_rule_application(&self, env: &mut Env<Self>, query: &mut Query) -> bool {
        // let mut query = query.clone();
        // self.canonize_application(&env, &mut query);
        // let memo_key = (env.clone(), query.clone());
        let mut canonized_env = env.clone();
        let mut canonized_query = query.clone();
        self.canonize_application_key(&mut canonized_env, &mut canonized_query);
        let memo_key = (canonized_env, canonized_query);

        if let Some((new_env, new_query)) = self.memoized_applications.get(&memo_key) {
            *env = new_env.clone();
            *query = new_query.clone();
            true
        } else {
            false
        }
    }

    fn save_solutions(&mut self, mut env: Env<Self>, mut query: Query, solutions: HashSet<Solution>) {
        self.canonize_solution_key(&mut env, &mut query);
        self.memoized_solutions.insert((env, query.clone()), solutions);
    }

    fn use_saved_solutions(&self, env: &Env<Self>, query: &Query) -> Option<HashSet<Solution>> {
        let (mut env, mut query) = (env.clone(), query.clone());
        self.canonize_solution_key(&mut env, &mut query);
        self.memoized_solutions.get(&(env, query)).cloned()
    }
}