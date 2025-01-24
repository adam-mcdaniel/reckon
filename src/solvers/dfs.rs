use std::collections::{BTreeMap, BTreeSet, HashMap};
use crate::{Query, Rule, Solution, Solver, Term, UnifyEnv, Var};
use tracing::*;

/// A simple DFS solver with memoization
#[derive(Default)]
pub struct DfsSolver {
    pub rules: Vec<Rule>,
    /// The memo table maps:
    /// (goals_after_substitution, sorted_bindings) -> Vec of solutions (each is a binding)
    pub memo: HashMap<(Vec<Term>, UnifyEnv), BTreeSet<Solution>>,
    pub max_recursion_limit: usize,
    pub full_solutions: BTreeSet<Solution>,
    pub steps: usize,
}

impl DfsSolver {
    pub fn new(mut rules: Vec<Rule>, max_recursion_limit: usize) -> Self {
        // Sort the rules by their size, so that we try the smallest rules first.
        // Put the recursive rules at the end.

        rules.sort_by_key(|r| r.size());
        let (mut recursive_rules, mut rules): (Vec<Rule>, Vec<Rule>) = rules.into_iter().partition(|r| r.is_recursive());

        rules.append(&mut recursive_rules);

        DfsSolver {
            rules,
            max_recursion_limit,
            ..Default::default()
        }
    }

    fn prove(
        &mut self,
        query: &Query,
        goals: &[Term],
        env: &UnifyEnv,
        max_solutions: usize,
        mut recursion_depth: usize,
        recursion_limit: usize,
        hit_recursion_limit: &mut bool,
    ) -> BTreeSet<Solution> {
        recursion_depth += 1;
        if recursion_depth > recursion_limit {
            *hit_recursion_limit = true;
            return BTreeSet::new();
        }

        // If no goals remain, we have a complete solution:
        // if goals.is_empty() || goals.iter().map(|g| g.reduce(env)).all(|g| matches!(g, Term::True)) {
        if goals.is_empty() || goals.iter().all(|g| matches!(g, Term::True)) {
            /*
            match env.to_full_solution(query) {
                Ok(sol) => {
                    self.solutions.insert(sol.clone());
                    return vec![sol];
                }
                Err(_) => {
                    return vec![];
                }
            }
            */
            if let Ok(sol) = env.to_full_solution(query) {
                // info!("Found full solution: {sol:?}");
                self.full_solutions.insert(sol.clone());
            } else {
                // info!("Found partial solution: {env:?}");
            }
            let mut solutions = BTreeSet::new();
            solutions.insert(env.to_partial_solution(query));
            return solutions;
        }

        // If the goals contains false
        if goals.iter().any(|g| matches!(g, Term::False)) {
            return BTreeSet::new();
        }

        // Apply current bindings to get canonical "substituted" goals
        let mut substituted_goals = goals.to_vec();
        for g in substituted_goals.iter_mut() {
            g.reduce_in_place(env);
        }

        // Sort the bindings into a canonical form for memoization
        // let mut sorted_bindings: Vec<(Var, Term)> =
        //     bindings_so_far.iter().map(|(k, v)| (*k, v.clone())).collect();
        // sorted_bindings.sort_by_key(|(k, _)| k.id);

        let memo_key = (substituted_goals.clone(), env.clone());
        if let Some(cached) = self.memo.get(&memo_key) {
            // Return a *clone* of the cached solutions
            return cached.clone();
        }

        // Take the first goal
        let mut first_goal = substituted_goals[0].clone();
        first_goal.reduce_in_place(env);
        let rest_goals = &substituted_goals[1..];

        // Special case: if the first goal is a Cut (!), we do not backtrack further:
        if matches!(first_goal, Term::Cut) {
            // Just solve the rest:
            let solutions_after_cut = self.prove(query, rest_goals, env, max_solutions, recursion_depth, recursion_limit, hit_recursion_limit);
            self.memo.insert(memo_key, solutions_after_cut.clone());
            return solutions_after_cut;
        }

        let mut all_solutions = BTreeSet::new();


        let mut env = env.clone();
        let mut tmp_env = env.clone();

        // for goal in goals {
        //     if let Ok(_) = goal.unify(&Term::True, &mut tmp_env) {
        //         env = tmp_env.clone();
        //     }
        // }
        if let Ok(_) = first_goal.unify(&Term::True, &mut tmp_env) {
            env = tmp_env.clone();
        }

        // Try each rule that might unify with this goal
        'outer: for r in &self.rules.clone() {
            if !r.might_apply_to(&first_goal) {
                // warn!("Skipping rule {:?} for goal {first_goal:?}", r);
                continue;
            }
            // info!("Trying rule {:?} for goal {first_goal:?}", r);
            self.steps += 1;
            let mut fresh_rule = r.clone();
            // refresh_rule_vars(&mut fresh_rule);
            fresh_rule.refresh();

            // --- KEY CHANGE: Do *not* substitute the rule head with `bindings_so_far`
            // before unify. Instead unify the goal (which is already substituted) 
            // against the *fresh* rule head:
            let mut new_env = env.clone();
            // Recurse
            // for goal in goals {
            //     if let Ok(_) = goal.unify(&Term::True, &mut tmp_env) {
            //         new_env = tmp_env.clone();
            //     }
            // }
            // If unification fails, skip to next rule
            if let Err(_) = first_goal.unify(&fresh_rule.head, &mut new_env) {
                continue;
            }

            // Now unify the tail with the updated `new_bindings`
            for tail_goal in fresh_rule.tail.iter_mut() {
                tail_goal.reduce_in_place(&new_env);
            }

            // Combine the newly substituted tail with the remaining goals
            let mut extended_goals = fresh_rule.tail.clone();
            extended_goals.extend_from_slice(rest_goals);

            extended_goals.iter_mut().for_each(|g| g.reduce_in_place(&new_env));

            // Recurse
            // for goal in extended_goals.iter_mut() {
            //     if let Ok(_) = goal.unify_in_place(&Term::True, &mut tmp_env) {
            //         new_env = tmp_env.clone();
            //     }
            // }

            let sub_solutions = self.prove(query, &extended_goals, &new_env, max_solutions, recursion_depth, recursion_limit, hit_recursion_limit);

            for sol in sub_solutions {
                all_solutions.insert(sol);
                if self.full_solutions.len() >= max_solutions {
                    break 'outer;
                }
            }
        }

        self.memo.insert(memo_key, all_solutions.clone());
        all_solutions
    }
}

impl Solver for DfsSolver {
    fn solve_vars(
        &mut self,
        query: &Query,
        max_solutions: usize,
    ) -> Result<BTreeSet<Solution>, String> {
        self.full_solutions.clear();
        self.steps = 0;

        // Check for invalid rules:
        for r in &self.rules {
            if r.is_invalid() {
                error!("Invalid rule: {:?}", r);
                return Err(format!("Invalid rule: {:?}", r));
            }
        }
        
        // We start with an empty binding environment
        let env = UnifyEnv::new(&self.rules, &[query.clone()]);
        let mut hit_recursion_limit = false;
        let mut recursion_limit = 50;

        let mut partial_solutions = BTreeSet::new();
        while self.full_solutions.len() < max_solutions {
            info!("Trying {query:?} with recursion limit: {}", recursion_limit);
            if recursion_limit > self.max_recursion_limit {
                break;
            }

            let mut current_partial_solutions = self.prove(query, &query.goals, &env, max_solutions, 0, recursion_limit, &mut hit_recursion_limit);
            partial_solutions.append(&mut current_partial_solutions);
            if hit_recursion_limit {
                recursion_limit *= 2;
                warn!("Hit recursion limit for query {query:?}, increasing to {recursion_limit}");
                hit_recursion_limit = false;
            } else {
                break;
            }
        }

        if self.full_solutions.is_empty() {
            if partial_solutions.is_empty() {
                return Err("No solutions found.".into());
            } else {
                warn!("Found {} partial solutions in {} steps", partial_solutions.len(), self.steps);
                Ok(partial_solutions)
            }
        } else {
            info!("Found {} solutions in {} steps", self.full_solutions.len(), self.steps);
            Ok(self.full_solutions.clone())
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_dfs_solver() {
        // Set up logging with the `tracing` crate, with info level logging.
        let _ = tracing_subscriber::fmt::SubscriberBuilder::default()
            .with_max_level(tracing::Level::INFO)
            .init();
        
        let prove_true = |rules: &[Rule], query: &str| {
            let mut solver = DfsSolver::new(rules.to_vec(), 100);
            let query = query.parse::<Query>().unwrap();
            let can_prove = solver.can_prove(&query);
            assert!(can_prove, "Failed to prove: {:?}", query);
        };

        let prove_false = |rules: &[Rule], query: &str| {
            let mut solver = DfsSolver::new(rules.to_vec(), 100);
            let query = query.parse::<Query>().unwrap();
            let can_prove = solver.can_prove(&query);
            assert!(!can_prove, "Proved: {:?}", query);
        };

        let find_solutions = |rules: &[Rule], query: &str, count: usize| {
            let mut solver = DfsSolver::new(rules.to_vec(), 100);
            let query = query.parse::<Query>().unwrap();
            let solutions = solver.solve_vars(&query, count).unwrap();

            info!("Solutions for:\n\n{query:?}\n\n{solutions:#?}\n\n");
        };

        let mut rules = vec![
            "not(A) :- ~A.".parse::<Rule>().unwrap(),
            "not(true) :- false.".parse::<Rule>().unwrap(),
            "not(false).".parse::<Rule>().unwrap(),
            "is_nat(0).".parse::<Rule>().unwrap(),
            "is_nat(s(X)) :- is_nat(X).".parse::<Rule>().unwrap(),
            "add(X, 0, X) :- is_nat(X).".parse::<Rule>().unwrap(),
            "add(X, s(Y), s(Z)) :- add(X, Y, Z).".parse::<Rule>().unwrap(),

            "mul(X, 0, 0) :- is_nat(X).".parse::<Rule>().unwrap(),
            "mul(X, s(Y), Z) :- mul(X, Y, W), add(X, W, Z).".parse::<Rule>().unwrap(),

            "leq(0, X, true) :- is_nat(X).".parse::<Rule>().unwrap(),
            "leq(s(X), s(Y), true) :- leq(X, Y, true).".parse::<Rule>().unwrap(),
            "leq(s(X), 0, false) :- is_nat(X).".parse::<Rule>().unwrap(),
            "leq(s(X), Y, false) :- leq(Y, X, true).".parse::<Rule>().unwrap(),

            "less(X, Y, false) :- leq(X, Y, true), leq(Y, X, true).".parse::<Rule>().unwrap(),
            "less(X, Y, true) :- leq(X, Y, true), leq(Y, X, false).".parse::<Rule>().unwrap(),

            "path(A, B) :- edge(A, B).".parse::<Rule>().unwrap(),
            "path(A, C) :- edge(A, B), path(B, C).".parse::<Rule>().unwrap(),

            "and(A, B) :- A, B.".parse::<Rule>().unwrap(),
            "or(A, B) :- A.".parse::<Rule>().unwrap(),
            "or(A, B) :- B.".parse::<Rule>().unwrap(),

            r#"edge("a", "b")."#.parse::<Rule>().unwrap(),
            r#"edge("b", "c")."#.parse::<Rule>().unwrap(),
            r#"edge("c", "d")."#.parse::<Rule>().unwrap(),
            r#"edge("d", "a")."#.parse::<Rule>().unwrap(),
        ];


        prove_true(&rules, "?- is_nat(0).");
        prove_true(&rules, "?- is_nat(s(0)).");
        prove_false(&rules, "?- is_nat(1).");
        find_solutions(&rules, "?- add(s(0), s(s(0)), A).", 5);
        find_solutions(&rules, "?- mul(A, B, s(s(s(s(0))))).", 2);
        find_solutions(&rules, "?- mul(A, B, A).", 10);
        prove_true(&rules, "?- mul(0, B, 0).");
        prove_true(&rules, "?- mul(s(0), B, B).");
        prove_true(&rules, r#"?- path("a", "a")."#);

        prove_false(&rules, "?- leq(s(s(s(s(0)))), s(s(s(0))), true).");
        prove_true(&rules, "?- leq(s(s(s(s(0)))), s(s(s(0))), false).");
        prove_true(&rules, "?- leq(s(s(s(0))), s(s(s(0))), true).");
        prove_true(&rules, "?- leq(s(s(0)), s(s(s(0))), true).");
        find_solutions(&rules, "?- leq(A, s(s(s(s(0)))), true).", 5);
        find_solutions(&rules, "?- leq(s(s(s(s(0)))), A, false).", 5);
        prove_false(&rules, "?- less(s(s(s(s(0)))), s(s(s(0))), true).");
        prove_false(&rules, "?- less(s(s(s(0))), s(s(s(0))), true).");
        prove_true(&rules, "?- less(s(s(0)), s(s(s(0))), true).");
        find_solutions(&rules, "?- less(A, s(s(s(0))), true).", 5);

        prove_true(&rules, "?- and(true, true).");
        prove_false(&rules, "?- and(true, false).");
        prove_false(&rules, "?- and(false, true).");
        prove_false(&rules, "?- and(false, false).");

        prove_true(&rules, "?- or(true, true).");
        prove_true(&rules, "?- or(true, false).");
        prove_true(&rules, "?- or(false, true).");
        prove_false(&rules, "?- or(false, false).");

        prove_true(&rules, "?- not(false).");
        prove_false(&rules, "?- not(true).");
        prove_true(&rules, "?- ^ [true, true].");
    }
}