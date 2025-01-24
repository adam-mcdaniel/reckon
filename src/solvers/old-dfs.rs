use std::collections::{BTreeMap, BTreeSet, HashMap};
use crate::{Term, Query, Rule, Solver, Var};
use tracing::*;

/// A simple DFS solver with memoization
pub struct DfsSolver {
    pub rules: Vec<Rule>,
    /// The memo table maps:
    /// (goals_after_substitution, sorted_bindings) -> Vec of solutions (each is a binding)
    pub memo: HashMap<(Vec<Term>, Vec<(Var, Term)>), Vec<BTreeMap<Var, Term>>>,
}

impl DfsSolver {
    pub fn new(rules: Vec<Rule>) -> Self {
        DfsSolver {
            rules,
            memo: HashMap::new(),
        }
    }

    /// The main entry point for proving a list of goals (with partial bindings).
    ///
    /// `goals` are the goals we still need to prove.
    /// `bindings_so_far` is the partial substitution already built up.
    /// Returns a list of new bindings that extend `bindings_so_far` for each solution.
    fn prove(
        &mut self,
        goals: &[Term],
        bindings_so_far: &BTreeMap<Var, Term>,
        max_solutions: usize,
        mut recursion_depth: usize,
    ) -> Vec<BTreeMap<Var, Term>> {
        // If there are no goals left, we've satisfied them all:
        if goals.is_empty() {
            return vec![bindings_so_far.clone()];
        }

        if recursion_depth > 20 {
            tracing::debug!("Recursion depth exceeded 100. Aborting.");
            return Vec::new();
        }
        recursion_depth += 1;

        // For memoization, we first apply the current partial bindings to the goals
        // to get a "canonical" set of goals.
        let mut substituted_goals: Vec<Term> = goals.to_vec();
        for goal in substituted_goals.iter_mut() {
            goal.substitute(bindings_so_far);
        }

        // Also produce a canonical representation of the current bindings
        // to use in the memo key. We'll just store them sorted by Var ID.
        let mut sorted_bindings: Vec<(Var, Term)> =
            bindings_so_far.iter().map(|(k, v)| (*k, v.clone())).collect();
        sorted_bindings.sort_by_key(|(k, _)| k.id);

        let memo_key = (substituted_goals.clone(), sorted_bindings.clone());
        if let Some(cached_solutions) = self.memo.get(&memo_key) {
            // Return a *copy* of the cached solutions
            return cached_solutions.clone();
        }

        // Otherwise, we need to prove the first goal
        let mut all_solutions = Vec::new();
        let mut rest_goals = substituted_goals[1..].to_vec();
        let first_goal = &substituted_goals[0];

        // Special case: cut (!) prunes all alternative solutions after it succeeds.
        // If the first goal is a cut, we succeed immediately with the current bindings,
        // and *do not* explore any alternatives.
        if matches!(first_goal, Term::Cut) {
            // Just prove the rest with no choice points left for this goal
            let solutions_after_cut = self.prove(&rest_goals, &bindings_so_far, max_solutions, recursion_depth);
            self.memo.insert(memo_key.clone(), solutions_after_cut.clone());
            return solutions_after_cut;
        }

        // For each rule in the KB, try to unify the rule head with the first goal
        'outer: for rule in &self.rules.clone() {
            // We'll "refresh" the variables in the rule so that each usage of the rule
            // has unique variable IDs.
            let mut fresh_rule = rule.clone();
            // The refresh method in this snippet is naive. Often you'd do a special rename
            // of the rule's variables to avoid collisions with goal variables.
            // refresh_rule_vars(&mut fresh_rule);
            fresh_rule.refresh(bindings_so_far);

            // Attempt to unify the rule head with the first goal
            // under the current bindings.
            let mut new_bindings = bindings_so_far.clone();

            // // We unify *after* substituting the head with current bindings
            // fresh_rule.head.substitute(&new_bindings);
            // // Attempt unify
            // if let Err(_) = fresh_rule.head.unify_with_bindings(first_goal, &mut new_bindings) {
            //     // Unification failed; move on to the next rule
            //     continue 'outer;
            // }
            // Remove this line:
            // Unify "first_goal" (already substituted by `bindings_so_far`) 
            // with the *fresh* rule head:
            if let Err(_) = first_goal.unify_with_bindings(&fresh_rule.head, &mut new_bindings) {
                continue 'outer;
            }

            // If unification succeeded, unify the rule's tail with the rest of the goals
            // The rule's tail may also reference the same variables, so we also
            // substitute them with `new_bindings` before combining.
            for tail_goal in fresh_rule.tail.iter_mut() {
                tail_goal.substitute(&new_bindings);
            }

            // Now the new goals to solve are fresh_rule.tail + the old rest_goals
            let mut extended_goals = fresh_rule.tail.clone();
            extended_goals.extend(rest_goals.clone());

            // Recurse
            let sub_solutions = self.prove(&extended_goals, &new_bindings, max_solutions, recursion_depth);

            // If we found solutions, accumulate them
            for s in sub_solutions {
                all_solutions.push(s);
                // If we have enough solutions, we can stop
                if all_solutions.len() >= max_solutions {
                    break 'outer;
                }
            }
        }

        // Memoize the solutions
        self.memo.insert(memo_key, all_solutions.clone());
        all_solutions
    }
}

impl Solver for DfsSolver {
    fn solve_vars(
        &mut self,
        query: &Query,
        max_solutions: usize,
    ) -> Result<Vec<BTreeMap<Var, Term>>, String> {
        // We start with an empty binding environment
        let bindings_so_far = BTreeMap::new();
        let mut solutions = self.prove(&query.goals, &bindings_so_far, max_solutions, 0);

        solutions = simplify_solutions(solutions);

        // Prune the solutions
        prune_solutions(&query.goals, &mut solutions);
        if solutions.is_empty() {
            Err("No solutions found.".into())
        } else {
            Ok(solutions)
        }
    }
}

/// A simple helper to refresh (rename) the variables in a Rule so that
/// each usage has distinct variable IDs. If your `Symbol` system already
/// gives fresh IDs for newly created variables, you can simply re-create them.
fn refresh_rule_vars(rule: &mut Rule) {
    // Gather all variables
    let mut vars_in_rule = BTreeSet::new();
    rule.head.free_vars(&mut vars_in_rule);
    for t in &rule.tail {
        t.free_vars(&mut vars_in_rule);
    }

    // For each var, make a new variable with a fresh symbol name
    let mut rename_map = BTreeMap::new();
    for old_var in vars_in_rule {
        let new_var = old_var.refresh();
        rename_map.insert(old_var, Term::Var(new_var));
    }

    // Substitute them in the rule
    rule.head.substitute(&rename_map);
    for t in rule.tail.iter_mut() {
        t.substitute(&rename_map);
    }
}



fn simplify_term(term: &Term, solution: &BTreeMap<Var, Term>) -> (Term, bool) {
    // Get the free variables of the goal
    let mut free_vars = BTreeSet::new();
    term.free_vars(&mut free_vars);

    let mut new_term = term.clone();

    // For every variable in the term to simplify,
    // replace it with the corresponding term in the solution.
    for var in free_vars {
        if let Some(substitute) = solution.get(&var) {
            new_term.substitute_var(var, substitute.clone());
        }
    }
    let is_simplified = new_term.has_free_vars();
    (new_term, is_simplified)
}

fn simplify_solution(solution: &BTreeMap<Var, Term>) -> BTreeMap<Var, Term> {
    let mut new_solution = BTreeMap::new();
    let mut all_simple = true;
    for _i in 0..100 {
        all_simple = true;
        for (var, term) in solution.iter() {
            let (new_term, is_simple) = simplify_term(term, &new_solution);
            new_solution.insert(*var, new_term);
            all_simple = all_simple && is_simple;
        }
        if all_simple {
            return new_solution;
        }
    }
    if !all_simple {
        debug!("Failed to simplify solution");
    }
    return new_solution;
}

fn simplify_solutions(solutions: Vec<BTreeMap<Var, Term>>) -> Vec<BTreeMap<Var, Term>> {
    let mut new_solutions = Vec::new();
    for solution in solutions.iter() {
        new_solutions.push(simplify_solution(solution));
    }
    new_solutions
}

fn prune_solution(goals: &[Term], solution: &mut BTreeMap<Var, Term>) {
    // Remove irrelevant bindings
    let mut relevant_vars = BTreeSet::new();
    for goal in goals.iter() {
        goal.free_vars(&mut relevant_vars);
    }
    
    solution.retain(|var, _| relevant_vars.contains(var));
}

fn prune_solutions(goals: &[Term], solutions: &mut Vec<BTreeMap<Var, Term>>) {
    for solution in solutions.iter_mut() {
        prune_solution(goals, solution);
    }
}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_naive_solver() {
        // Set up logging with the `tracing` crate, with info level logging.
        let _ = tracing_subscriber::fmt::SubscriberBuilder::default()
            .with_max_level(tracing::Level::INFO)
            .init();
        

        let rules = vec![
            "is_nat(0).".parse::<Rule>().unwrap(),
            "is_nat(s(X)) :- is_nat(X).".parse::<Rule>().unwrap(),
            "add(X, 0, X) :- is_nat(X).".parse::<Rule>().unwrap(),
            "add(X, s(Y), s(Z)) :- add(X, Y, Z).".parse::<Rule>().unwrap(),

            "mul(X, 0, 0) :- is_nat(X).".parse::<Rule>().unwrap(),
            "mul(X, s(Y), Z) :- mul(X, Y, W), add(X, W, Z).".parse::<Rule>().unwrap(),
        ];

        println!("Rules: {:#?}", rules);

        let mut solver = DfsSolver::new(rules);

        let query = "?- mul(A, A, s(s(s(s(0))))).".parse::<Query>().unwrap();
        let solutions = solver.solve_vars(&query, 10).unwrap();

        println!("Solutions: {:#?}", solutions);
        println!("Can prove?: {}", solver.can_prove(&query));
    }
}