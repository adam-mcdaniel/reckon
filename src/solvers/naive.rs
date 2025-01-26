use std::collections::{BTreeMap, HashSet, HashMap};  // <-- add HashMap
use std::hash::{Hash, Hasher};                        use crate::Env;
// <-- needed for hashing
use crate::{solvers::{Solver}, term, Query, Rule, Term, Var};
use tracing::{info, warn, debug, error};

// A key type for memo cache
// We'll store the goals and the current bindings.  Because `BTreeMap<Var, Term>`
// doesn't directly implement `Hash` out of the box, we can flatten it into a vector.
// Alternatively, you could implement a custom Hash for (Vec<Term>, BTreeMap<Var,Term>).
#[derive(Clone, Eq, PartialEq)]
struct MemoKey {
    goals: Vec<Term>,
    bindings: Vec<(Var, Term)>,
}

// We need a custom Hash implementation that ensures
// that `(goals, bindings)` produce a unique, order-independent hash
// for the `bindings`. Because it's a BTreeMap, we know it's sorted by key.
impl Hash for MemoKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the goals
        for g in &self.goals {
            g.hash(state);
        }
        // Hash the (Var, Term) pairs in `bindings` in order
        for (var, term) in &self.bindings {
            var.hash(state);
            term.hash(state);
        }
    }
}

impl MemoKey {
    fn new(goals: &[Term], current_bindings: &BTreeMap<Var, Term>) -> Self {
        // Flatten the BTreeMap<Var,Term> into a Vec<(Var,Term)> for hashing/comparison.
        // This preserves the order of keys (BTreeMap is already sorted).
        let bindings_vec = current_bindings.iter().map(|(k,v)| (*k, v.clone())).collect();
        MemoKey {
            goals: goals.to_vec(),
            bindings: bindings_vec,
        }
    }
}

/// A naive backtracking solver that tries each rule in turn, unifying the current goal with the rule head.
/// When unification succeeds, it appends the rule tail to the remaining goals and recurses.
/// Now includes memoization to avoid recomputing solutions for the same sub-problems.
pub struct NaiveSolver {
    pub rules: Vec<Rule>,

    // NEW: add a cache for memoization.
    // We'll store, for a given (subgoal, current_bindings), which solutions were found.
    memo: HashMap<MemoKey, HashSet<BTreeMap<Var, Term>>>,
}

impl NaiveSolver {
    /// Create a new `NaiveSolver` with a list of rules.
    pub fn new(rules: Vec<Rule>) -> Self {
        NaiveSolver {
            rules,
            memo: HashMap::new(),  // initialize empty
        }
    }

    fn next_goals(&self, rule: &Rule, current_bindings: &BTreeMap<Var, Term>, rest_goals: &[Term]) -> Vec<Term> {
        let mut next_goals = Vec::new();
        next_goals.extend_from_slice(&rule.tail);
        next_goals.extend_from_slice(rest_goals);

        // Substitute current_bindings into each of these new goals
        for g in next_goals.iter_mut() {
            g.substitute(current_bindings);
        }

        next_goals
    }

    /// Recursively solve the list of goals, backtracking over rule choices.
    /// Returns a set of all successful binding maps (one per solution).
    fn solve_goals(
        &mut self,
        overall_goals: &[Term],
        goals: &[Term],
        env: Env,
        recursive_depth: usize,
        solutions_limit: &mut isize,
        unique_solutions: &mut HashSet<Env>,
    ) {
        // 1. If no goals remain, we've satisfied them all. Add the current bindings as a solution.
        if goals.is_empty() {
            // Simplify the solutions
            // let mut cb = simplify_solution(&current_bindings);

            // Prune the solutions
            // prune_solution(overall_goals, &mut cb);
            unique_solutions.insert(env);
            return;
        }

        if recursive_depth > 20 {
            debug!("Recursion depth exceeded 100. Aborting.");
            return;
        }

        // -- NEW: Before we do any real work, check the memo cache.
        let key = MemoKey::new(goals, &current_bindings);
        if let Some(cached) = self.memo.get(&key) {
            // We already computed solutions for these `(goals, current_bindings)`.
            // Just add them into `unique_solutions` and return.
            for sol in cached {
                unique_solutions.insert(sol.clone());
            }
            return;
        }

        // We'll accumulate solutions locally, then store them in memo at the end.
        let mut local_solutions = HashSet::new();

        // 2. Otherwise, take the first goal and attempt to solve it.
        let (first_goal, rest_goals) = goals.split_first().unwrap();

        // 3. Handle the built-in cut (!) specially.
        if let Term::Cut = first_goal {
            self.solve_goals(
                overall_goals,
                rest_goals,
                current_bindings,
                recursive_depth + 1,
                solutions_limit,
                &mut local_solutions,
            );
            // Merge local solutions into overall solutions
            for sol in &local_solutions {
                unique_solutions.insert(sol.clone());
            }
            // Store in memo, then return.
            self.memo.insert(key, local_solutions);
            return;
        }

        // 4. Try matching this goal against each rule in the knowledge base.
        for rule in self.rules.clone() {
            let mut rule = rule.clone();
            rule.refresh();

            let mut new_bindings = current_bindings.clone();
            if first_goal
                .unify_in_place(&rule.head, &mut )
                .is_ok()
            {
                debug!(
                    "Unification successful: {:?} with {:?}",
                    first_goal, rule.head
                );
                // Next goals = rule.tail + remaining goals
                let mut next_goals = self.next_goals(&rule, &new_bindings, rest_goals);

                // Sort the next goals by the solution score
                next_goals.sort_by_key(|goal| goal.size());
                next_goals.reverse();

                // Recurse
                self.solve_goals(
                    overall_goals,
                    &next_goals,
                    new_bindings,
                    recursive_depth + 1,
                    solutions_limit,
                    &mut local_solutions
                );

                if local_solutions.len() as isize >= *solutions_limit {
                    warn!("Solution limit reached. Stopping.");
                    break;
                }
            } else {
                debug!("Unification failed: {:?} with {:?}", first_goal, rule.head);
            }
        }

        // Merge local solutions into the overall solutions.
        for sol in &local_solutions {
            unique_solutions.insert(sol.clone());
        }

        // -- NEW: store the sub-problem solutions into the memo cache.
        self.memo.insert(key, local_solutions);
    }
}

// Helper functions remain mostly the same...
fn simplify_term(term: &Term, solution: &BTreeMap<Var, Term>) -> (Term, bool) {
    let mut used_vars = HashSet::new();
    term.used_vars(&mut used_vars);

    let mut new_term = term.clone();
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
    new_solution
}

fn prune_solution(goals: &[Term], solution: &mut BTreeMap<Var, Term>) {
    let mut relevant_vars = HashSet::new();
    for goal in goals.iter() {
        goal.used_vars(&mut relevant_vars);
    }
    solution.retain(|var, _| relevant_vars.contains(var));
}

impl Solver for NaiveSolver {
    /// Solve a query and return a list of variable bindings for each solution found.
    fn solve_vars(
        &mut self,
        query: &Query,
        max_solutions: usize
    ) -> Result<HashSet<BTreeMap<Var, Term>>, String> {
        let mut unique_solutions = HashSet::new();

        // Clear or reuse the memo. If you want to reuse the memo
        // across different queries, skip clearing it.
        // self.memo.clear();

        self.solve_goals(
            &query.goals,
            &query.goals,
            BTreeMap::new(),
            0,
            &mut (max_solutions as isize),
            &mut unique_solutions
        );

        if unique_solutions.is_empty() {
            return Err("No solutions found: the query is unprovable.".to_string());
        }
        Ok(unique_solutions)
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

        let mut solver = NaiveSolver::new(rules);

        let query = "?- mul(A, B, s(s(s(s(0))))).".parse::<Query>().unwrap();
        let solutions = solver.solve_vars(&query, 10).unwrap();

        println!("Solutions: {:#?}", solutions);
        println!("Can prove?: {}", solver.can_prove(&query));
    }
}