use super::*;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use tracing::{debug, warn};
use std::fmt::{Display, Debug, Formatter, Result as FmtResult};
use std::hash::Hash;


/// A search traversal strategy.
/// 
/// This enum represents the different strategies that can be used to traverse the search space.
/// 
/// - `DepthFirst`: Traverse the search space in a depth-first manner. This will pursue a single path as far as possible before backtracking.
/// - `BreadthFirst`: Traverse the search space in a breadth-first manner. This will explore all paths at a given depth before moving to the next depth.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Traversal {
    DepthFirst,
    #[default]
    BreadthFirst,
}

impl Traversal {
    /// Get the next item from the queue, depending on the traversal strategy.
    pub fn get_next_in_queue<T>(&mut self, queue: &mut VecDeque<T>) -> Option<T> {
        match self {
            Traversal::DepthFirst => queue.pop_back(),
            Traversal::BreadthFirst => queue.pop_front(),
        }
    }

    /// Push an item to the queue, depending on the traversal strategy.
    pub fn push_to_queue<T>(&self, queue: &mut VecDeque<T>, item: T) {
        match self {
            Traversal::DepthFirst => queue.push_back(item),
            Traversal::BreadthFirst => queue.push_back(item),
        }
    }
}

/// A configuration for a search.
/// 
/// This struct contains all the parameters that can be used to configure the search for solutions.
/// 
/// - `max_search_depth`: The maximum depth to search to. If `None`, the search will continue until a solution is found or the step limit is reached.
/// 
/// - `max_search_width`: The maximum width of the search -- this constrains how many subsearches can be performed for each rule application.
/// If the width is reached, the possible paths remaining for a given query will be pruned.
/// 
/// - `step_limit`: The maximum number of steps to perform in the search. If `None`, the search will continue until a solution is found or the depth limit is reached.
/// 
/// - `prune`: Whether to prune the search space by removing redundant variables and simplifying queries while searching. This likely decreases the search space,
/// but may increase the time taken to find a solution.
/// 
/// - `traversal`: The traversal strategy to use for the search.
/// 
/// - `require_head_match`: Whether to require that a rule applies to the first goal in a query before applying it to the rest of the goals.
/// 
/// - `queue_sorter`: A function that can be used to sort the queue of queries after a certain number of steps. This can be used to prioritize certain queries.
/// 
/// - `sort_after_steps`: The number of steps to perform before sorting the queue of queries.
/// 
/// - `reduce_query`: Whether to reduce the query after each rule application. This can be used to simplify the query and reduce the search space, but may increase the time taken to find a solution.
/// 
/// - `solution_limit`: The maximum number of solutions to find. If this limit is reached, the search will stop.
/// 
/// - `clean_memoization`: Whether to clean the memoization cache after the full search. After the full search is performed,
/// the final solution is retained and memoized, but the sub-solutions are removed from the cache.
#[derive(Clone)]
pub struct SearchConfig<S> where S: Solver {
    /// The maximum depth to search to. If `None`, the search will continue until a solution is found or the step limit is reached.
    pub max_search_depth: Option<usize>,
    /// The maximum width of the search -- this constrains how many subsearches can be performed for each rule application.
    pub max_search_width: Option<usize>,
    /// The maximum number of steps to perform in the search. If `None`, the search will continue until a solution is found or the depth limit is reached.
    pub step_limit: Option<usize>,
    /// Whether to prune the search space by removing redundant variables and simplifying queries while searching. This likely decreases the search space,
    pub prune: bool,
    /// The traversal strategy to use for the search.
    pub traversal: Traversal,
    /// Whether to require that a rule applies to the first goal in a query before applying it to the rest of the goals.
    pub require_head_match: bool,
    /// A function that can be used to sort the queue of queries after a certain number of steps. This can be used to prioritize certain queries.
    pub queue_sorter: Option<Arc<dyn Fn(&mut VecDeque<(Env<S>, Query, usize)>)>>,
    /// The number of steps to perform before sorting the queue of queries.
    pub sort_after_steps: usize,
    /// Whether to reduce the query after each rule application. This can be used to simplify the query and reduce the search space, but may increase the time taken to find a solution.
    pub reduce_query: bool,
    /// The maximum number of solutions to find. If this limit is reached, the search will stop.
    pub solution_limit: usize,
    /// Whether to clean the memoization cache after the full search. After the full search is performed,
    pub clean_memoization: bool,
    /// Stop application after the first goal in the query
    pub stop_after_first_goal: bool,
}

/// Print the search configuration.
impl<S> Debug for SearchConfig<S> where S: Solver {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "SearchConfig")
    }
}

/// Compare two search configurations for equality.
/// This is just to allow environments to be compared for equality.
impl<S> PartialEq for SearchConfig<S> where S: Solver {
    fn eq(&self, other: &Self) -> bool {
        self.max_search_depth == other.max_search_depth
            && self.max_search_width == other.max_search_width
            && self.step_limit == other.step_limit
            && self.prune == other.prune
            && self.traversal == other.traversal
    }
}

/// Compare two search configurations for equality.
impl<S> Eq for SearchConfig<S> where S: Solver {}


impl<S> SearchConfig<S> where S: Solver {
    /// Create a new search configuration with permissive defaults.
    /// 
    /// This configuration allows for a wide search space, with no limits on depth, width, or steps.
    /// 
    /// It is not recommended to use this configuration for most searches, as it may lead to long search times and high memory usage.
    /// 
    /// If you use this configuration, you should set limits on the search depth, width, and steps.
    pub fn permissive() -> Self {
        SearchConfig {
            max_search_depth: None,
            max_search_width: None,
            step_limit: None,
            prune: false,
            require_head_match: false,
            traversal: Traversal::BreadthFirst,
            queue_sorter: None,
            sort_after_steps: 1,
            reduce_query: false,
            solution_limit: 1,
            clean_memoization: false,
            stop_after_first_goal: false,
        }
    }

    /// Parse a search configuration from a string.
    /// 
    /// This will parse the new options from the string and update the search configuration.
    pub fn parse<'a, 'b>(&'a mut self, s: &'b str) -> Result<&'b str, String> {
        parse_search_config(s, self)
            .map_err(|e| match e {
                nom::Err::Error(e) | nom::Err::Failure(e) => nom::error::convert_error(s, e),
                nom::Err::Incomplete(_) => "Incomplete input".to_string(),
            }).map(|(rest, _)| rest)
    }

    /// Create a new search configuration with the given maximum search depth.
    pub fn with_solution_limit(mut self, solution_limit: usize) -> Self {
        self.solution_limit = solution_limit;
        self
    }

    /// Create a new search configuration with the given traversal strategy.
    pub fn with_traversal(mut self, traversal: Traversal) -> Self {
        self.traversal = traversal;
        self
    }

    /// Create a new search configuration with the given clean memoization setting.
    pub fn with_clean_memoization(mut self, clean_memoization: bool) -> Self {
        self.clean_memoization = clean_memoization;
        self
    }

    /// Create a new search configuration with the given stop after first goal setting.
    pub fn with_stop_after_first_goal(mut self, stop_after_first_goal: bool) -> Self {
        self.stop_after_first_goal = stop_after_first_goal;
        self
    }

    /// Create a new search configuration with the given reduce query setting.
    pub fn with_reduce_query(mut self, reduce_query: bool) -> Self {
        self.reduce_query = reduce_query;
        self
    }

    /// Create a new search configuration with the given require head match setting.
    pub fn with_require_rule_head_match(mut self, require_head_match: bool) -> Self {
        self.require_head_match = require_head_match;
        self
    }
    
    /// Create a new search configuration with the given depth limit.
    pub fn with_depth_limit(mut self, max_search_depth: usize) -> Self {
        self.max_search_depth = Some(max_search_depth);
        self
    }

    /// Create a new search configuration with the given width limit.
    pub fn with_width_limit(mut self, max_search_width: usize) -> Self {
        self.max_search_width = Some(max_search_width);
        self
    }

    /// Create a new search configuration with the given step limit.
    pub fn with_step_limit(mut self, step_limit: usize) -> Self {
        self.step_limit = Some(step_limit);
        self
    }

    /// Create a new search configuration with the given sorter function, and the number of steps to sort after.
    pub fn with_sorter<K>(mut self, after_steps: usize, sorter: impl Fn(&Env<S>, &Query) -> K + 'static) -> Self where K: Ord {
        self.sort_after_steps = after_steps;
        self.queue_sorter = Some(Arc::new(
            move |queue: &mut VecDeque<(Env<S>, Query, usize)>| {
                let queue = queue.make_contiguous();
                queue.sort_by_key(|(env, query, _)| {
                    sorter(env, query)
                });

                match self.traversal {
                    Traversal::DepthFirst => queue.reverse(),
                    Traversal::BreadthFirst => (),
                }
            }
        ));
        self
    }

    /// Sort the queue of queries using the sorter function.
    pub fn sort_queue(&self, queue: &mut VecDeque<(Env<S>, Query, usize)>) {
        if let Some(sorter) = &self.queue_sorter {
            debug!("Sorting queue");
            sorter(queue);
        }
    }

    /// Create a new search configuration with the given pruning setting.
    pub fn with_pruning(mut self, prune: bool) -> Self {
        self.prune = prune;
        self
    }

    /// Check if the search configuration allows pruning.
    pub fn can_prune(&self) -> bool {
        self.prune
    }

    /// Check if the search configuration allows deeper searches.
    pub fn can_search_deeper(&self, depth: usize) -> bool {
        self.max_search_depth.map(|max_depth| depth < max_depth).unwrap_or(true)
    }

    /// Check if the search configuration allows wider searches.
    pub fn can_search_wider(&self, width: usize) -> bool {
        self.max_search_width.map(|max_width| width < max_width).unwrap_or(true)
    }

    /// Check if the search configuration allows more steps to be performed.
    pub fn can_perform_step(&self, steps: usize) -> bool {
        self.step_limit.map(|limit| steps < limit).unwrap_or(true)
    }
}

impl<S> Default for SearchConfig<S> where S: Solver {
    fn default() -> Self {
        SearchConfig::permissive()
            .with_depth_limit(300)
            .with_require_rule_head_match(true)
            .with_width_limit(5)
            .with_clean_memoization(true)
    }
}

/// An environment for a solver.
/// 
/// This stores the rules, variable bindings, search configuration, and solver for a given search.
/// 
/// The environment is used to store the state of the search, store variable bindings for unification,
/// and to perform the search for solutions.
/// 
/// It also stores some bookkeeping information, such as the number of steps performed in the search.
#[derive(Default, Debug, Clone, Eq)]
pub struct Env<S> where S: Solver {
    /// The rules to use in the search.
    rules: Arc<Vec<Rule>>,
    /// The variable bindings for the search.
    var_bindings: Arc<HashMap<Var, Term>>,
    /// The solver to use for the search.
    search_config: SearchConfig<S>,
    /// The solver to use for the search. This allows for different memoization strategies to be used
    /// by supplying different solvers.
    solver: S,
    /// The queries that have been proven false.
    /// This is used to avoid re-proving the same queries multiple times.
    proven_false: HashSet<Query>,
    /// The number of steps performed in the search.
    steps: usize,
}

/// Compare two environments for equality.
impl<S> PartialEq for Env<S> where S: Solver {
    fn eq(&self, other: &Self) -> bool {
        self.rules == other.rules && self.var_bindings == other.var_bindings
    }
}

/// Hash the environment to allow it to be used in hash maps.
/// 
/// This hashes the rules and variable bindings.
/// 
/// Because the environment uses a hashmap for the variable bindings, the variables are sorted before hashing.
impl<S> Hash for Env<S> where S: Solver {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.rules.hash(state);
        // Sort the variables
        let mut var_bindings: Vec<_> = self.var_bindings.iter().collect();
        var_bindings.sort_by(|(var1, _), (var2, _)| var1.cmp(var2));

        for (var, term) in var_bindings {
            var.hash(state);
            term.hash(state);
        }
    }
}

impl<S> Env<S> where S: Solver {
    /// Create a new environment with the given rules.
    pub fn new(rules: &[Rule]) -> Self {
        Env {
            rules: Arc::new(rules.to_vec()),
            var_bindings: Arc::new(HashMap::new()),
            solver: S::default(),
            search_config: SearchConfig::default(),
            proven_false: HashSet::new(),
            steps: 0,
        }
    }

    /// Reset the solver in the environment.
    /// 
    /// This will clear the memoization cache in the solver.
    pub fn reset_solver(&mut self) {
        self.solver.reset();
    }

    /// Clear the environment, removing all rules and variable bindings.
    pub fn reset(&mut self) {
        self.var_bindings = Arc::new(HashMap::new());
        self.proven_false = HashSet::new();
        self.steps = 0;
        self.solver.reset();
    }

    /// Get the number of steps performed in the search.
    pub fn get_steps(&self) -> usize {
        self.steps
    }

    /// Reset the number of steps performed in the search.
    pub fn reset_steps(&mut self) {
        self.steps = 0;
    }

    /// Get the search configuration for the environment.
    pub fn search_config_mut(&mut self) -> &mut SearchConfig<S> {
        &mut self.search_config
    }
    
    /// Supply a new search configuration for the environment.
    pub fn with_search_config(mut self, search_config: &SearchConfig<S>) -> Self {
        self.search_config = search_config.clone();
        self
    }

    /// Get the variables used in the environment.
    pub fn used_vars(&self, vars: &mut HashSet<Var>) {
        for term in self.var_bindings.values() {
            term.used_vars(vars);
        }
    }

    /// Remove useless variables.
    /// 
    /// If X is bound to Y, and Y is bound to Z, then X is bound to Z.
    pub fn prune_redundant_variables(&mut self) -> HashMap<Var, Term> {
        if !self.search_config.can_prune() {
            warn!("Pruning is disabled");
            return HashMap::new();
        }

        let mut new_var_bindings = (*self.var_bindings).clone();
        let mut pruned_bindings = HashMap::new();
        let mut changed = true;
        while changed {
            changed = false;
            for (var, mut term) in new_var_bindings.clone() {
                let old_term = term.clone();
                term.substitute(&new_var_bindings);
                if term != old_term {
                    pruned_bindings.insert(var, term.clone());
                    new_var_bindings.insert(var, term);
                    changed = true;
                }
            }
        }


        if !pruned_bindings.is_empty() {
            // warn!("Pruning variables:");
            // for (var, term) in &pruned_bindings {
            //     warn!("{} = {} (was {})", var, term, self.var_bindings.get(var).unwrap());
            // }
            *self.var_bindings_mut() = new_var_bindings;
            pruned_bindings
        } else {
            HashMap::new()
        }
    }

    /// Prune the environment, removing unused variables.
    /// 
    /// It takes the full, original query supplied by the user, and the current query that is being searched.
    pub fn prune(&mut self, original_query: &Query, query: &mut Query) {
        if !self.search_config.can_prune() {
            warn!("Pruning is disabled");
            return;
        }

        let new_vars = self.prune_redundant_variables();
        query.substitute(&new_vars);
        // warn!("Original query: {original_query}");
        // warn!("Pruned query: {query}");
        // Remove unused variables from the bindings
        let mut used_vars = HashSet::new();
        query.used_vars(&mut used_vars);
        original_query.used_vars(&mut used_vars);
        // warn!("Unpruned query: {query}");

        // For every var in the query, add their used variables to the set
        let mut changed = true;
        while changed {
            changed = false;
            for var in used_vars.clone() {
                if let Some(term) = self.var_bindings.get(&var) {
                    let old_len = used_vars.len();
                    term.used_vars(&mut used_vars);
                    changed = changed || old_len != used_vars.len();
                }
            }
        }
        
        let mut unused_vars: Vec<_> = self.var_bindings.keys().cloned().collect();
        unused_vars.retain(|var| !used_vars.contains(var));
        // warn!("Unused variables:");
        // for var in unused_vars {
        //     warn!("{}", var);
        // }

        self.var_bindings_mut().retain(|var, _| used_vars.contains(var));

        // warn!("Pruned environment:");
        // for (var, term) in self.var_bindings.iter() {
        //     warn!("{} = {}", var, term);
        // }
    }

    /// Get the variable bindings for the environment.
    pub fn var_bindings(&self) -> &HashMap<Var, Term> {
        &self.var_bindings
    }

    /// Get the mutable variable bindings for the environment.
    pub fn var_bindings_mut(&mut self) -> &mut HashMap<Var, Term> {
        Arc::make_mut(&mut self.var_bindings)
    }

    /// Set a variable in the environment.
    pub fn set_var(&mut self, var: Var, term: Term) {
        Arc::make_mut(&mut self.var_bindings).insert(var, term);
    }

    /// Get a variable from the environment.
    pub fn get_var(&self, var: Var) -> Option<&Term> {
        self.var_bindings.get(&var)
    }

    /// Get the rules for the environment.
    pub fn get_rules(&self) -> &[Rule] {
        &self.rules
    }

    /// Add a rule to the environment.
    pub fn add_rule(&mut self, rule: Rule) {
        Arc::make_mut(&mut self.rules).push(rule);
    }

    /// Convert the bindings in this environment to a solution.
    /// 
    /// This will take all the variables in the query, and return the reduced bindings for those variables.
    /// 
    /// If there are variables in the query that are not bound, this will return an error with the set of unbound variables.
    pub fn to_full_solution(&self, original_query: &Query, final_query: &Query) -> Result<Solution, HashSet<Term>> {
        // Filter out the free variables that are not in the query
        let mut free_query_vars = HashSet::new();
        original_query.used_vars(&mut free_query_vars);

        let mut partial_solution = self.to_partial_solution(original_query, final_query);
        
        if !final_query.is_ground_truth() {
            return Err(partial_solution.substitute_into_query(final_query).goals().cloned().collect());
        }

        partial_solution.var_bindings.retain(|var, _| free_query_vars.contains(var));

        if partial_solution.var_bindings.len() < free_query_vars.len() {
            return Err(
                free_query_vars.difference(&partial_solution.var_bindings.keys().cloned().collect()).cloned()
                    .map(|var| Term::Var(var))
                    .collect()
            );
        }

        Ok(partial_solution)
    }

    /// Convert the bindings in this environment to a partial solution.
    /// 
    /// This will take all the variables in the query, and return the reduced bindings for those variables.
    /// 
    /// If all the variables are bound, this is a full solution.
    /// If there are variables in the query that are not bound, this will return the partial solution with
    /// as many variables bound as possible.
    pub fn to_partial_solution(&self, original_query: &Query, final_query: &Query) -> Solution {
        let mut free_query_vars = HashSet::new();
        original_query.used_vars(&mut free_query_vars);
        // Now that we have the free variables from the query, 
        // simplify them in the bindings until there are no more free variables
        let mut bindings = (*self.var_bindings).clone();
        let mut has_found_used_vars;
        for _ in 0..100 {
            has_found_used_vars = false;
            
            bindings = bindings.iter().map(|(var, term)| {
                let mut term = term.clone();
                term.substitute(&bindings);
                (var.clone(), term)
            }).collect();

            for free_query_var in &free_query_vars {
                if let Some(term) = bindings.get(free_query_var) {
                    has_found_used_vars = has_found_used_vars || term.has_used_vars();
                }
                if has_found_used_vars {
                    break;
                }
            }
            if !has_found_used_vars {
                break;
            }
        }
        
        Solution::new(original_query.clone(), final_query.clone(), bindings)
    }

    /// Apply all the rules in the environment to the goal term, adding the new goals to the query.
    pub fn apply_rules(&mut self, goal: &Term, query: &mut Query) -> bool {
        let mut changed = false;
        for i in 0..self.rules.len() {
            changed = changed || self.apply_rule(i, goal, query);
        }
        changed
    }

    /// Apply all the rules in the environment to all the goals query, adding the new goals to the query.
    /// 
    /// If the search configuration requires a head match,
    /// the rules will be applied to the first goal in the query before applying them to the rest of the goals.
    /// If the application fails, the rest of the goals will not be considered.
    /// 
    /// If the search configuration does not require a head match, the rules will always be applied to all the goals in the query.
    pub fn apply_rules_to_query(&mut self, query: &mut Query) -> bool {
        let mut new_query = query.clone();
        let mut changed = false;
        for (i, term) in query.goals().enumerate() {
            changed = changed || self.apply_rules(term, &mut new_query);
            if i == 0 && !changed && self.search_config.require_head_match {
                debug!("Bailing out of rule application, head match required");
                return false;
            }

            if self.search_config.stop_after_first_goal && changed {
                break;
            }
        }
        *query = new_query;
        changed
    }

    
    /// Apply a rule to a goal term, adding the new goals to the query.
    pub fn apply_rule(&mut self, rule: usize, goal: &Term, query: &mut Query) -> bool {
        let rule = self.rules[rule].clone();
        // debug!("Applying rule {} to goal {} in query {}", rule, goal, query);
        if self.search_config.can_prune() {
            let new_bindings = self.prune_redundant_variables();
            query.substitute(&new_bindings);
        }

        // let mut old_env = self.clone();
        // let old_query = query.clone();
        // if self.solver.use_saved_rule_application(&mut old_env, query) {
        //     *self = old_env;
        //     debug!("Skipping rule application, using memoized result for {rule} applied to {goal} in {query}");
        //     return true;
        // }
        let changed = rule.apply(goal, query, self);
        // self.solver.save_rule_application(old_env, &old_query, self.clone(), query);
        changed
    }

    /// Apply a rule to a query, adding the new goals to the query.
    /// 
    /// If the search configuration requires a head match,
    /// the rule will be applied to the first goal in the query before applying it to the rest of the goals.
    /// If the application fails, the rest of the goals will not be considered.
    /// 
    /// If the search configuration does not require a head match, the rule will always be applied to all the goals in the query.
    pub fn apply_rule_to_query(&mut self, rule: usize, query: &mut Query) -> bool {
        // let terms = query.goals().cloned().collect::<Vec<Term>>();
        let mut new_query = query.clone();
        let mut changed = false;
        for (i, term) in query.goals().enumerate() {
            changed = changed || self.apply_rule(rule, &term, &mut new_query);
            if i == 0 && !changed && self.search_config.require_head_match {
                debug!("Bailing out of rule application, head match required");
                return false;
            }

            if self.search_config.stop_after_first_goal && changed {
                break;
            }
        }
        *query = new_query;
        changed
    }

    /// Attempt to find a solution for the query using the search configuration.
    /// 
    /// If a solution is found, it will be returned.
    /// If no solution is found, an error will be returned with the set of unbound variables.
    pub fn prove_true(&mut self, query: &Query) -> Result<Solution, HashSet<Term>> {
        // let solutions = self.find_solutions_dfs(&query, &query, 1, 0, &mut steps);
        // let solutions = self.find_solutions_bfs(&query, &query, 1);
        let old_solution_limit = self.search_config.solution_limit;
        self.search_config.solution_limit = 1;
        let solutions = self.find_solutions(&query);
        self.search_config.solution_limit = old_solution_limit;

        if let Ok(mut solutions) = solutions {
            if let Some(solution) = solutions.drain().next() {
                return Ok(solution);
            }
        }

        return Err(query.goals().cloned().collect());

        /*
        while !query.is_ground_truth() && !query.contains_contradiction() {
            query.remove_provable_complements(self);
            if !self.apply_rules_to_query(&mut query) {
                error!("Could not apply any rules to query: {}", old_query);
                return Err(query.goals().cloned().collect());
            }
        }

        if query.is_ground_truth() {
            debug!("Query is true: {}", old_query);
            return self.to_full_solution(&old_query, &query);
        }

        if query.contains_contradiction() {
            error!("Query contains contradiction: {}", old_query);
            return Err(query.goals().cloned().collect());
        }

        self.to_full_solution(&old_query, &query)
         */
    }

    /// Attempt to prove that the query is false using the search configuration.
    /// 
    /// If the query is false, it will be added to the set of proven false queries,
    /// and the function will return `Ok(())`.
    /// 
    /// If the query is true, an error will be returned with the set of unsolved goals.
    pub fn prove_false(&mut self, query: &Query) -> Result<(), HashSet<Term>> {
        debug!("Proving false: {}", query);
        if self.proven_false.contains(query) {
            return Ok(());
        }
        
        if self.prove_true(query).is_ok() {
            debug!("Query is true, not false: {}", query);
            Err(query.goals().cloned().collect())
        } else {
            debug!("Query is false: {}", query);
            self.proven_false.insert(query.clone());
            Ok(())
        }
    }

    /*
    pub fn find_solutions(&mut self, original_query: &Query, count: usize) -> Result<HashSet<Solution>, HashSet<Term>> {
        let mut solutions = HashSet::new();
        // Instead of doing a BFS, we'll do a DFS
        let mut queue: VecDeque<_> = VecDeque::new();
        queue.push_back((self.clone(), original_query.clone(), 0));

        // let mut last_checkpoint = (self.clone(), original_query.clone());
        let mut iterations_without_sorting = 0;
        while let Some((mut env, mut next_query, depth)) = self.search_config.traversal.get_next_in_queue(&mut queue) {
            // last_checkpoint = (env.clone(), next_query.clone());
            if solutions.len() >= count || !self.search_config.can_perform_step(self.steps) {
                break;
            }

            if !self.search_config.can_search_deeper(depth) {
                // warn!("Recursion limit reached, pruning query: {}", old_query);
                continue;
            }

            next_query.reduce_in_place(&env);
            if self.search_config.prune {
                env.prune(original_query, &mut next_query);
            }

            if next_query.is_ground_truth() {
                let solution = env.to_full_solution(&original_query, &next_query)?;
                solutions.insert(solution.clone());
                debug!("Found solution #{}: {}", solutions.len(), solution);
                self.solver.save_solutions(env, next_query, solutions.clone());
                continue;
            }

            if next_query.contains_contradiction() {
                warn!("Query contains contradiction: {}", next_query);
                continue;
            }

            let mut current_search_width = 0;
            for i in 0..self.rules.len() {
                if !self.search_config.can_search_wider(current_search_width) {
                    debug!("Width limit reached, pruning query: {}", next_query);
                    break;
                }
                
                let mut tmp_env = env.clone();
                let mut tmp_query = next_query.clone();

                if tmp_env.apply_rule_to_query(i, &mut tmp_query) {
                    tmp_query.remove_provable_complements(&mut tmp_env);

                    self.steps += 1;
                    if self.steps % 1000 == 0 {
                        debug!("Step {}: {next_query}", self.steps);
                    }

                    if self.search_config.can_search_deeper(depth + 1) {
                        self.search_config.traversal.push_to_queue(&mut queue, (tmp_env, tmp_query, depth + 1));
                        iterations_without_sorting += 1;
                        current_search_width += 1;
                    }
                }
            }

            if iterations_without_sorting > self.search_config.sort_after_steps {
                self.search_config.sort_queue(&mut queue);
                iterations_without_sorting = 0;
            }
        }

        if queue.is_empty() {
            debug!("DFS finished, no more available query paths");
        } else {
            debug!("Stopped search after {} steps", self.steps);
        }
        if !self.search_config.can_perform_step(self.steps) {
            debug!("DFS finished, reached step limit after {} steps", self.steps);
        }

        if solutions.is_empty() {
            // error!("Could not find solution for query: {}", last_checkpoint.1);
            // error!("In env: {:#?}", last_checkpoint.0);
            // error!("Last checkpoint: env: {env:#?}\nquery: {old_query}");
            Err(original_query.goals().cloned().collect())
            // Err(last_checkpoint.1.goals().cloned().collect())
        } else {
            debug!("Solved in {} steps", self.steps);
            Ok(solutions)
        }
    }
     */

    /// Find solutions for the query using the given search configuration.
    pub fn find_solutions(&mut self, original_query: &Query) -> Result<HashSet<Solution>, HashSet<Term>> {
        match self.search_config.traversal {
            Traversal::DepthFirst => {
                self.find_solutions_dfs(original_query, self.search_config.solution_limit)
            }
            Traversal::BreadthFirst => {
                self.find_solutions_bfs(original_query, self.search_config.solution_limit)
            }
        }
    }

    /// Perform a DFS search for solutions, using the given search configuration.
    pub fn find_solutions_dfs(
        &mut self,
        original_query: &Query,
        max_solutions: usize,
    ) -> Result<HashSet<Solution>, HashSet<Term>> {
        // Use a queue of states. Each state will be a tuple containing:
        // (environment, current query, depth, current search width).
        let mut queue = VecDeque::new();

        // We’ll track solutions in a HashSet to avoid duplicates.
        let mut solutions = HashSet::new();

        // Keep track of total steps.
        let mut total_steps = 0;

        // Initialize the queue with the starting state.
        // Depth = 0, current search width = 0 (or 1, depending on how you interpret width).
        queue.push_back((self.clone(), original_query.clone(), 0_usize));
        let mut iterations_without_sorting = 0;

        while let Some((mut env, query, depth)) = queue.pop_back() {
            if iterations_without_sorting > self.search_config.sort_after_steps {
                self.search_config.sort_queue(&mut queue);
                iterations_without_sorting = 0;
            }

            // -- Check if we've already found enough solutions --
            if solutions.len() >= max_solutions {
                debug!(
                    "Reached max_solutions ({}) solutions, stopping BFS.",
                    max_solutions
                );
                break;
            }

            // -- Depth limit check (if your config enforces it) --
            if !env.search_config.can_search_deeper(depth) {
                warn!("Depth limit reached for query: {}", query);
                continue;
            }

            // -- Memoization check --
            if let Some(memoized_solutions) = self.solver.use_saved_solutions(&env, &query) {
                debug!("Using memoized solutions for query: {}", query);
                for sol in memoized_solutions {
                    solutions.insert(sol);
                }
                // Continue BFS; maybe you want to skip expansions here or not.
                continue;
            }
            
            // -- Ground truth check --
            if query.is_ground_truth() {
                let solution = match env.to_full_solution(original_query, &query) {
                    Ok(sol) => sol,
                    Err(e) => {
                        // If we cannot derive a solution for whatever reason, skip.
                        warn!("Error deriving full solution: {:?}", e);
                        continue;
                    }
                };

                // debug!("Found solution #{}: {}", solutions.len() + 1, solution);
                solutions.insert(solution.clone());

                // Save solutions using the memoization layer (if appropriate).
                self.solver.save_solutions(env.clone(), query.clone(), solutions.clone());
                // Continue BFS to find more solutions (or break if you only need the first).
                continue;
            }

            // -- Contradiction check --
            if query.contains_contradiction() {
                // warn!("Query contains contradiction: {}", query);
                continue;
            }

            // -- Simplify query / environment, handle pruning, etc. --
            // let mut simplified_query = query.reduce(&env);
            let mut simplified_query = if self.search_config.reduce_query {
                query.reduce(&env)
            } else {
                query.clone()
            };
            let mut simplified_env = env.clone();
            if simplified_env.search_config.prune {
                simplified_env.prune(original_query, &mut simplified_query);
            }
            simplified_query.remove_irreducible_negatives_in_place(&mut simplified_env);
            // if simplified_query.remove_provable_complements(&mut simplified_env) {
            //     error!("Absurdity detected in query: {}", simplified_query);
            //     continue;
            // }

            // We will apply each rule in the environment to the simplified query,
            // generating new states for our BFS frontier.
            let mut width_used = 0_usize;
            for i in 0..env.rules.len() {
                // -- Check if we've already found enough solutions --
                if solutions.len() >= max_solutions {
                    debug!("Found enough solutions, breaking out of rule loop.");
                    break;
                }

                // -- Check if we can search wider (based on search width config) --
                if !env.search_config.can_search_wider(width_used) {
                    debug!("Width limit reached, pruning query: {}", simplified_query);
                    break;
                }

                // -- Clone environment and query for rule application --
                let mut tmp_env = simplified_env.clone();
                let mut tmp_query = simplified_query.clone();

                // Try applying the i-th rule:
                if tmp_env.apply_rule_to_query(i, &mut tmp_query) {
                    tmp_query.remove_irreducible_negatives_in_place(&mut tmp_env);
                    // Additional complements removal, pruning, etc.
                    if tmp_env.search_config.prune {
                        tmp_env.prune(original_query, &mut tmp_query);
                    }

                    if total_steps % 100 == 0 {
                        debug!("Step {} (queue size={}): {tmp_query}", total_steps, queue.len());
                        // if tmp_query.remove_provable_complements(&mut tmp_env) {
                        //     error!("Absurdity detected in query: {}", tmp_query);
                        //     continue;
                        // }
                    }

                    total_steps += 1;
                    width_used += 1;
                    iterations_without_sorting += 1;
                    // -- Push the resulting state back onto the BFS queue --
                    queue.push_back((tmp_env, tmp_query, depth + 1));
                } else {
                    debug!(
                        "Could not apply rule {} to query: {}",
                        env.rules[i], tmp_query
                    );
                }
            }
            // Track the total steps in the main environment as well.
            env.steps = total_steps;

            if !solutions.is_empty() {
                // debug!(
                //     "Found {} solutions for query after {} BFS expansions: {}",
                //     solutions.len(),
                //     total_steps,
                //     query
                // );
            }

            // -- Save solutions if any were found at this node --
            self.solver.save_solutions(env, query, solutions.clone());
        } // end of while queue is not empty

        if self.search_config.clean_memoization {
            self.solver.reset();
        }
        self.solver.save_solutions(self.clone(), original_query.clone(), solutions.clone());
        self.steps = total_steps;

        if !solutions.is_empty() {
            debug!("Found {} solutions for query after {} steps: {}", solutions.len(), self.steps, original_query);
            Ok(solutions)
        } else {
            // If BFS ends without finding solutions, return an empty set or an error
            Ok(HashSet::new())
        }
    }

    /// Perform a BFS search for solutions, using the given search configuration.
    pub fn find_solutions_bfs(
        &mut self,
        original_query: &Query,
        max_solutions: usize,
    ) -> Result<HashSet<Solution>, HashSet<Term>> {
        // Use a queue of states. Each state will be a tuple containing:
        // (environment, current query, depth, current search width).
        let mut queue = VecDeque::new();

        // We’ll track solutions in a HashSet to avoid duplicates.
        let mut solutions = HashSet::new();

        // Keep track of total steps.
        let mut total_steps = 0;

        // Initialize the queue with the starting state.
        // Depth = 0, current search width = 0 (or 1, depending on how you interpret width).
        queue.push_back((self.clone(), original_query.clone(), 0_usize));
        let mut iterations_without_sorting = 0;

        while let Some((mut env, query, depth)) = queue.pop_front() {
            if iterations_without_sorting > self.search_config.sort_after_steps {
                self.search_config.sort_queue(&mut queue);
                iterations_without_sorting = 0;
            }

            // -- Check if we've already found enough solutions --
            if solutions.len() >= max_solutions {
                debug!(
                    "Reached max_solutions ({}) solutions, stopping BFS.",
                    max_solutions
                );
                break;
            }


            // -- Depth limit check (if your config enforces it) --
            if !env.search_config.can_search_deeper(depth) {
                warn!("Depth limit reached for query: {}", query);
                continue;
            }

            // -- Memoization check --
            if let Some(memoized_solutions) = self.solver.use_saved_solutions(&env, &query) {
                debug!("Using memoized solutions for query: {}", query);
                for sol in memoized_solutions {
                    solutions.insert(sol);
                }
                // Continue BFS; maybe you want to skip expansions here or not.
                continue;
            }

            // -- Ground truth check --
            if query.is_ground_truth() {
                let solution = match env.to_full_solution(original_query, &query) {
                    Ok(sol) => sol,
                    Err(e) => {
                        // If we cannot derive a solution for whatever reason, skip.
                        warn!("Error deriving full solution: {:?}", e);
                        continue;
                    }
                };

                // debug!("Found solution #{}: {}", solutions.len() + 1, solution);
                solutions.insert(solution.clone());

                // Save solutions using the memoization layer (if appropriate).
                self.solver.save_solutions(env.clone(), query.clone(), solutions.clone());
                // Continue BFS to find more solutions (or break if you only need the first).
                continue;
            }
            
            // -- Contradiction check --
            if query.contains_contradiction() {
                // warn!("Query contains contradiction: {}", query);
                continue;
            }

            // -- Simplify query / environment, handle pruning, etc. --
            // let mut simplified_query = query.reduce(&env);
            // let mut simplified_query = query.clone();
            let mut simplified_query = if self.search_config.reduce_query {
                query.reduce(&env)
            } else {
                query.clone()
            };
            let mut simplified_env = env.clone();
            if simplified_env.search_config.prune {
                simplified_env.prune(original_query, &mut simplified_query);
            }
            simplified_query.remove_irreducible_negatives_in_place(&mut simplified_env);
            // if simplified_query.remove_provable_complements(&mut simplified_env) {
            //     error!("Absurdity detected in query: {}", simplified_query);
            //     continue;
            // }

            // We will apply each rule in the environment to the simplified query,
            // generating new states for our BFS frontier.
            let mut width_used = 0_usize;
            for i in 0..env.rules.len() {
                // -- Check if we've already found enough solutions --
                if solutions.len() >= max_solutions {
                    debug!("Found enough solutions, breaking out of rule loop.");
                    break;
                }

                // -- Check if we can search wider (based on search width config) --
                if !env.search_config.can_search_wider(width_used) {
                    debug!("Width limit reached, pruning query: {}", simplified_query);
                    break;
                }

                // -- Clone environment and query for rule application --
                let mut tmp_env = simplified_env.clone();
                let mut tmp_query = simplified_query.clone();

                // Try applying the i-th rule:
                if tmp_env.apply_rule_to_query(i, &mut tmp_query) {
                    tmp_query.remove_irreducible_negatives_in_place(&mut tmp_env);
                    // Additional complements removal, pruning, etc.
                    if tmp_env.search_config.prune {
                        tmp_env.prune(original_query, &mut tmp_query);
                    }

                    if total_steps % 100 == 0 {
                        debug!("Step {} (queue size={}): {tmp_query}", total_steps, queue.len());
                        // if tmp_query.remove_provable_complements(&mut tmp_env) {
                        //     error!("Absurdity detected in query: {}", tmp_query);
                        //     continue;
                        // }
                    }

                    total_steps += 1;
                    width_used += 1;
                    iterations_without_sorting += 1;
                    // -- Push the resulting state back onto the BFS queue --
                    queue.push_back((tmp_env, tmp_query, depth + 1));
                } else {
                    debug!(
                        "Could not apply rule {} to query: {}",
                        env.rules[i], tmp_query
                    );
                }
            }
            // Track the total steps in the main environment as well.
            env.steps = total_steps;

            if !solutions.is_empty() {
                // debug!(
                //     "Found {} solutions for query after {} BFS expansions: {}",
                //     solutions.len(),
                //     total_steps,
                //     query
                // );
            }
            
            // -- Save solutions if any were found at this node --
            self.solver.save_solutions(env, query, solutions.clone());
        } // end of while queue is not empty
        if self.search_config.clean_memoization {
            self.solver.reset();
        }
        self.solver.save_solutions(self.clone(), original_query.clone(), solutions.clone());
        self.steps = total_steps;

        if !solutions.is_empty() {
            debug!("Found {} solutions for query after {} steps: {}", solutions.len(), self.steps, original_query);
            Ok(solutions)
        } else {
            // If BFS ends without finding solutions, return an empty set or an error
            Ok(HashSet::new())
        }
    }
}


/// A solution to a query.
/// 
/// This struct contains the original query, the final proved query, and the variable bindings.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Solution {
    /// The original query that was given to the solver.
    pub original_query: Query,
    /// The final query that was proved by the solver.
    pub final_query: Query,
    /// The variable bindings that were found by the solver.
    pub var_bindings: BTreeMap<Var, Term>,
}

impl Solution {
    /// Create a new solution with the given original query, final query, and variable bindings.
    pub fn new(original_query: Query, final_query: Query, var_bindings: impl IntoIterator<Item=(Var, Term)>) -> Self {
        Solution { original_query, final_query, var_bindings: var_bindings.into_iter().collect() }
    }

    /// Check if the solution is complete, i.e., if the final query is a ground truth (contains no variables).
    pub fn is_complete(&self) -> bool {
        self.final_query.is_ground_truth()
    }

    /// Substitute the variable bindings into a query.
    fn substitute_into_query(&self, query: &Query) -> Query {
        let mut new_query = query.clone();
        new_query.substitute(&self.var_bindings.clone().into_iter().collect());
        new_query
    }

    /// Get the variable bindings for the solution.
    pub fn var_bindings(&self) -> &BTreeMap<Var, Term> {
        &self.var_bindings
    }
}

impl<S> From<Solution> for Env<S> where S: Solver {
    fn from(solution: Solution) -> Self {
        Env {
            search_config: SearchConfig::default(),
            rules: Arc::new(vec![]),
            var_bindings: Arc::new(solution.var_bindings.into_iter().collect()),
            solver: S::default(),
            proven_false: HashSet::new(),
            steps: 0,
        }
    }
}

impl<S> Display for Env<S> where S: Solver {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        if self.rules.is_empty() {
            write!(f, "No rules\n")?;
        } else {
            write!(f, "Rules:\n")?;
            for (i, rule) in self.rules.iter().enumerate() {
                write!(f, "{}. {}\n", i+1, rule)?;
            }
        }
        if !self.var_bindings.is_empty() {
            write!(f, "Variable bindings:\n")?;
            for (var, term) in &*self.var_bindings {
                write!(f, "{} = {}\n", var, term)?;
            }
        } else {
            write!(f, "No variable bindings\n")?;
        }
        Ok(())
    }
}
impl Display for Solution {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "Solution for query: {}\n", self.original_query)?;
        write!(f, "Final query: {}\n", self.final_query)?;
        write!(f, "Variable bindings:\n")?;
        for (var, term) in &self.var_bindings {
            write!(f, "{} = {}\n", var, term)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use tracing::{info, error};

    fn time_it<F: FnMut() -> R, R>(mut f: F) -> (R, std::time::Duration) {
        let start = std::time::Instant::now();
        let result = f();
        let duration = start.elapsed();
        (result, duration)
    }

    #[test]
    fn test_prove_multiplication() {
        let rules: Vec<Rule> = vec![
            "is_nat(s(X)) :- is_nat(X).".parse().unwrap(),
            "is_nat(0).".parse().unwrap(),
            "add(X, s(Y), s(Z)) :- add(X, Y, Z).".parse().unwrap(),
            "add(X, 0, X) :- is_nat(X).".parse().unwrap(),
            "leq(0, X) :- is_nat(X).".parse().unwrap(),
            "leq(s(X), s(Y)) :- leq(X, Y).".parse().unwrap(),
            
            "geq(X, Y) :- leq(Y, X).".parse().unwrap(),
            "eq(X, Y) :- leq(X, Y), leq(Y, X).".parse().unwrap(),
            "neq(X, Y) :- ~eq(X, Y).".parse().unwrap(),
            
            
            "mul(X, s(Y), Z) :- mul(X, Y, W), add(X, W, Z).".parse().unwrap(),
            "mul(X, 0, 0) :- is_nat(X).".parse().unwrap(),
            "square(X, Y) :- mul(X, X, Y).".parse().unwrap(),
            // "isprime(X, Y, Z) :- is_nat(X), is_nat(Y), is_nat(Z), ~eq(X, s(0)), ~eq(Y, s(0)),  ~eq(X, 0), ~eq(Y, 0), ~eq(Z, 0), ~mul(X, Y, Z).".parse().unwrap(),
        ];

        // let query: Query = "?- add(A, B, s(s(s(s(0))))).".parse().unwrap();
        let mut n = String::from("0");
        for _ in 0..4 {
            n = format!("s({})", n);
        }

        // let query: Query = format!("?- ~add(A, A, s(s(s(s(0))))), add(A, B, C), leq(A, s(s(s(0)))), leq(B, s(s(s(0)))), leq(C, s(s(s(s(0))))).").parse().unwrap();
        // let mut query: Query = format!(r#"?- neq(A, s(0)), mul(A, B, {n})."#).parse().unwrap();
        let query: Query = format!(r#"?- mul({n}, {n}, A)."#).parse().unwrap();
            
        // let query: Query = "?- neq(A, s(0)), neq(B, s(0)), mul(A, B, s(s(s(s(0))))).".parse().unwrap();
        // let query: Query = "?- isprime(X, Y, s(s(s(0)))).".parse().unwrap();

        let mut env = Env::<DefaultSolver>::new(&rules)
            .with_search_config(
                &SearchConfig::default()
                    // .with_step_limit(1000)
                    .with_traversal(Traversal::BreadthFirst)
                    // .with_traversal(Traversal::DepthFirst)
                    .with_depth_limit(500)
                    // .with_width_limit(5)
                    .with_pruning(false)
                    .with_sorter(100, |_, query: &Query| query.size())
                    // .with_sorter(100, |_, query: &Query| usize::MAX - query.size())
                    .with_require_rule_head_match(true)
                    .with_reduce_query(false)
            );

        // println!("{}", query);
        // println!("{}", env.prove_true(&mut query).unwrap());
        // return;

        // match env.prove_true(&query) {
        //     Ok(solution) => {
        //         println!("{}", solution);
        //     }
        //     Err(terms) => {
        //         error!("Could not find solution for query: ");
        //         for term in terms {
        //             error!("{}", term);
        //         }
        //     }
        // }
        let (solutions, duration) = time_it(|| {
            env.find_solutions(&query)
        });
        // let solutions = env.find_solutions(&query, 5);

        match solutions {
            Ok(solutions) => {
                info!("Found {} solutions in {:?}", solutions.len(), duration);
                for (i, solution) in solutions.iter().enumerate() {
                    info!("Solution #{}: ", i + 1);
                    for (var, term) in solution.var_bindings() {
                        info!("{} = {}", var, term);
                    }
                }
            },
            Err(terms) => {
                error!("Could not find solution for query: ");
                for term in terms {
                    error!("{}", term);
                }
            }
        }
    }


    #[test]
    fn test_application() {
        let rules: Vec<Rule> = vec![
            "f(X) :- g(X).".parse().unwrap(),
            "g(X) :- h(X).".parse().unwrap(),
            "h(X) :- i(X).".parse().unwrap(),
            "i(1).".parse().unwrap(),
        ];

        let mut query: Query = "?- f(X).".parse().unwrap();
        let old_query = query.clone();

        let mut env = Env::<DefaultSolver>::new(&rules);
        println!("{:?}", query);

        env.apply_rules_to_query(&mut query);

        println!("{:?}", query);
        println!("{env:#?}");
        env.apply_rules_to_query(&mut query);

        println!("{:?}", query);
        println!("{env:#?}");
        
        env.apply_rules_to_query(&mut query);

        println!("{:?}", query);
        println!("{env:#?}");
        
        env.apply_rules_to_query(&mut query);

        println!("{:?}", query);
        println!("{env:#?}");

        let solution = env.to_full_solution(&old_query, &query);
        println!("{:?}", solution);


        let mut query: Query = "?- f(X), f(2).".parse().unwrap();
        env.apply_rules_to_query(&mut query);
    }


    #[test]
    fn test_proof() {
        let rules: Vec<Rule> = vec![
            "f(X) :- g(X).".parse().unwrap(),
            "g(X) :- h(X).".parse().unwrap(),
            "h(X) :- i(X).".parse().unwrap(),
            "i(1).".parse().unwrap(),
        ];

        let query: Query = "?- f(1).".parse().unwrap();

        let mut env = Env::<DefaultSolver>::new(&rules);
        println!("{:?}", query);

        let solution = env.prove_true(&query);
        println!("{:?}", solution);
    }

    #[test]
    fn test_prove_addition() {
        let rules: Vec<Rule> = vec![
            "is_nat(0).".parse().unwrap(),
            "is_nat(s(X)) :- is_nat(X).".parse().unwrap(),
            "add(X, 0, X) :- is_nat(X).".parse().unwrap(),
            "add(X, s(Y), s(Z)) :- add(X, Y, Z).".parse().unwrap(),
        ];

        let query: Query = "?- add(s(s(0)), s(0), X).".parse().unwrap();

        let mut env = Env::<DefaultSolver>::new(&rules);
        println!("{:?}", query);

        let solution = env.prove_true(&query);

        match solution {
            Ok(solution) => {
                println!("{}", solution);
            }
            Err(terms) => {
                error!("Could not find solution for query: ");
                for term in terms {
                    error!("{}", term);
                }
            }
        }
    }
}