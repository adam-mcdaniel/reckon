use std::collections::{BTreeMap, HashSet};

use crate::{solvers::{Solver, SolutionScorer, DefaultSolutionScorer}, term, Query, Rule, Term, Var};
use tracing::{info, warn, debug, error};

/// A naive backtracking solver that tries each rule in turn, unifying the current goal with the rule head.
/// When unification succeeds, it appends the rule tail to the remaining goals and recurses.
pub struct NaiveSolver {
    pub rules: Vec<Rule>,
}

impl NaiveSolver {
    /// Create a new `NaiveSolver` with a list of rules.
    pub fn new(rules: Vec<Rule>) -> Self {
        NaiveSolver { rules }
    }

    fn next_goals(&self, rule: &Rule, current_bindings: &BTreeMap<Var, Term>, rest_goals: &[Term]) -> Vec<Term> {
        let mut next_goals = Vec::new();
        next_goals.extend_from_slice(&rule.tail);
        next_goals.extend_from_slice(rest_goals);

        // Substitute new_bindings into each of these new goals, so they're more ground.
        for g in next_goals.iter_mut() {
            g.substitute(current_bindings);
        }

        next_goals
    }

    /// Recursively solve the list of goals, backtracking over rule choices.
    /// Returns a set of all successful binding maps (one per solution).
    fn solve_goals(
        &self,
        overall_goals: &[Term],
        goals: &[Term],
        mut current_bindings: BTreeMap<Var, Term>,
        scorer: &impl SolutionScorer,
        recursive_depth: usize,
        mut solutions_limit: &mut isize,
        unique_solutions: &mut HashSet<BTreeMap<Var, Term>>,
    ) {
        // 1. If no goals remain, we've satisfied them all. Add the current bindings as a solution.
        if goals.is_empty() {
            // Simplify the solutions
            current_bindings = simplify_solution(&current_bindings);
    
            // Prune the solutions
            prune_solution(overall_goals, &mut current_bindings);
            unique_solutions.insert(current_bindings);
            return;
        }

        if recursive_depth > 100 {
            debug!("Recursion depth exceeded 100. Aborting.");
            return;
        }

        // 2. Otherwise, take the first goal and attempt to solve it.
        let (first_goal, rest_goals) = goals.split_first().unwrap();

        // 3. Handle the built-in cut (!) specially.
        //    If the first goal is a cut, prune other alternatives and continue with the rest.
        if let Term::Cut = first_goal {
            self.solve_goals(overall_goals, rest_goals, current_bindings, scorer, recursive_depth + 1, solutions_limit, unique_solutions);
            return;
        }

        // 4. Try matching this goal against each rule in the knowledge base.
        for rule in &self.rules {
            let mut rule = rule.clone();
            rule.refresh(&current_bindings);
            // We'll unify the first goal with this rule's head.
            // Make a clone of the current bindings to avoid overwriting them in case of failure.
            let mut new_bindings = current_bindings.clone();

            // Attempt to unify `first_goal` with `rule.head`.
            if first_goal
                .unify_with_bindings(&rule.head, &mut new_bindings)
                .is_ok()
            {
                debug!(
                    "Unification successful: {:?} with {:?}",
                    first_goal, rule.head
                );
                // If unification is successful, we now have updated `new_bindings`.
                // Next goals = (rule.tail) + (remaining goals).
                // We must apply the updated bindings to the tail + rest of the goals before recursing.
                let mut next_goals = self.next_goals(&rule, &new_bindings, rest_goals);

                // Sort the next goals by the solution score.
                next_goals.sort_by_key(|goal| {
                    scorer.score(goal, &new_bindings)
                });

                // 5. Recursively solve the resulting goals. Gather all sub-solutions.
                self.solve_goals(overall_goals, &next_goals, new_bindings, scorer, recursive_depth + 1, solutions_limit, unique_solutions);
                if unique_solutions.len() as isize >= *solutions_limit {
                    warn!("Solution limit reached. Stopping.");
                    break;
                }
            } else {
                debug!(
                    "Unification failed: {:?} with {:?}",
                    first_goal, rule.head
                );
            }
        }
    }
}

fn simplify_term(term: &Term, solution: &BTreeMap<Var, Term>) -> (Term, bool) {
    // Get the free variables of the goal
    let mut used_vars = HashSet::new();
    term.used_vars(&mut used_vars);

    let mut new_term = term.clone();

    // For every variable in the term to simplify,
    // replace it with the corresponding term in the solution.
    for var in used_vars {
        if let Some(substitute) = solution.get(&var) {
            new_term.substitute_var(var, substitute.clone());
        }
    }
    let is_simplified = new_term.has_used_vars();
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
    let mut relevant_vars = HashSet::new();
    for goal in goals.iter() {
        goal.used_vars(&mut relevant_vars);
    }
    
    solution.retain(|var, _| relevant_vars.contains(var));
}

fn prune_solutions(goals: &[Term], solutions: &mut Vec<BTreeMap<Var, Term>>) {
    for solution in solutions.iter_mut() {
        prune_solution(goals, solution);
    }
}

impl Solver for NaiveSolver {
    /// Solve a query and return a list of variable bindings for each solution found.
    /// Returns an error if no solutions are found (i.e., the query is unprovable).
    fn solve_vars(&mut self, query: &Query, scorer: &impl SolutionScorer, max_solutions: usize) -> Result<Vec<BTreeMap<Var, Term>>, String> {
        // Start with empty bindings.
        let mut unique_solutions = HashSet::new();
        self.solve_goals(&query.goals, &query.goals, BTreeMap::new(), scorer, 0, &mut (max_solutions as isize), &mut unique_solutions);

        // If no solutions are found, return an error.
        if unique_solutions.is_empty() {
            return Err("No solutions found: the query is unprovable.".to_string());
        }

        // Otherwise, return the solutions.
        Ok(unique_solutions.into_iter().collect())
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
            "mul(s(X), Y, Z) :- mul(X, Y, W), add(Y, W, Z).".parse::<Rule>().unwrap(),
        ];

        println!("Rules: {:#?}", rules);

        let mut solver = NaiveSolver::new(rules);

        let query = "?- mul(A, B, s(s(s(s(0))))).".parse::<Query>().unwrap();
        let solutions = solver.solve_vars(&query, &DefaultSolutionScorer, 10).unwrap();

        println!("Solutions: {:#?}", solutions);
        println!("Can prove?: {}", solver.can_prove(&query, &DefaultSolutionScorer));
    }
}