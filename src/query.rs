use super::*;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::ops::Index;
use std::str::FromStr;
use std::sync::Arc;
use std::fmt::{Debug, Display, Formatter, Result as FmtResult};
use tracing::{debug, error, info};

/// A query to provide to the solver to prove true or false.
/// 
/// A query is a list of goals that the solver must prove true.
/// If the solver can prove all the goals true, then the query is true.
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Query {
    /// The goals to prove true
    goals: Arc<BTreeSet<Term>>,
}

impl Query {
    /// Create a new query with the given goals to prove
    pub fn new(goals: impl IntoIterator<Item = Term>) -> Self {
        Query { goals: Arc::new(goals.into_iter().collect()) }
    }

    /// Create an empty query, which is interpreted as true
    pub fn empty() -> Self {
        Query::new(vec![])
    }

    /// Get the sum of the sizes of all the goals in the query
    /// 
    /// The size of a goal term is the total number of subterms in the term
    pub fn size(&self) -> usize {
        self.goals.iter().map(|goal| goal.size()).sum()
    }

    /// Remove all the complemented goals that cannot be simplified
    /// by any of the rules in the environment.
    /// 
    /// These are inherently negative goals that cannot be proven
    pub fn remove_irreducible_negatives_in_place(&mut self, env: &Env<impl Solver>) {
        if !self.has_complements() {
            return;
        }
        self.goals = self.remove_irreducible_negatives(env).goals;
    }

    /// Does the query contain any complements?
    pub fn has_complements(&self) -> bool {
        self.goals.iter().any(|goal| match goal {
            Term::Complement(_) => true,
            _ => false,
        })
    }

    /// Does the query contain only complements?
    /// 
    /// If there are no complements, this returns false
    pub fn has_only_complements(&self) -> bool {
        self.goals.iter().all(|goal| match goal {
            Term::Complement(_) => true,
            _ => false,
        })
    }

    /// Filter out all the complements, returning only the positive goals
    pub fn filter_out_complements(&self) -> Self {
        Query::new(self.goals.iter().filter(|goal| match goal {
            Term::Complement(_) => false,
            _ => true,
        }).cloned())
    }

    /// Merge the goals of this query with the goals of another query,
    /// returning a new query with all the goals
    pub fn join(&self, other: &Query) -> Self {
        Query::new(self.goals.iter().chain(other.goals.iter()).cloned())
    }

    /// Remove all the complements that we can prove are true.
    /// 
    /// This is done by finding all the complements, and then trying to prove
    /// that the complement is false. If we can prove that the complement is false,
    /// then we can remove the complement from the query.
    /// 
    /// If we prove a complement as absurd, then we can return true, indicating that
    /// the query is absurd.
    pub fn remove_provable_complements(&mut self, env: &mut Env<impl Solver>) -> bool {
        // if !self.has_complements() {
        if !self.has_complements() {
            return false;
        }
        self.remove_irreducible_negatives_in_place(&env);
        let original = self.clone();

        let mut new_goals = BTreeSet::new();
        for goal in self.goals() {
            match goal {
                Term::Complement(term) => {
                    let query = Query::new([*term.clone()]);


                    // let mut query = self.clone();
                    // query.remove_goal(goal);
                    // query.add_positive_goal(*term.clone());
                    
                    let mut tmp_env = env.clone();
                    if tmp_env.prove_true(&query).is_ok() {
                        // Proved the complement is true, so absurdity was found
                        error!("Could prove {} is true, so absurdity was found", goal);
                        return true;
                    } else {
                        info!("Could prove {} is true, so removing it", goal);
                        *env = tmp_env;
                    }

                    // let mut tmp_env = env.clone();
                    // if tmp_env.prove_false(&mut query).is_ok() {
                    //     *env = tmp_env;
                    // } else {
                    //     // info!("Could not prove {} is false, so keeping it", term);
                    //     new_goals.insert(goal.clone());
                    // }
                }
                _ => {
                    new_goals.insert(goal.clone());
                }
            }
        }
        self.goals = Arc::new(new_goals);
        info!("After removing provable complements: {} (was {original})", self);
        false
    }


    /// Remove all the complemented goals that cannot be simplified
    /// by any of the rules in the environment.
    /// 
    /// These are inherently negative goals that cannot be proven.
    /// 
    /// This is a non-mutating version of `remove_irreducible_negatives_in_place`
    pub fn remove_irreducible_negatives(&self, env: &Env<impl Solver>) -> Self {
        if !self.has_complements() {
            return self.clone();
        }

        let mut result = BTreeSet::new();
        for goal in self.goals() {
            match goal {
                Term::Complement(term) => {
                    // Check if there are any free variables in the term
                    let mut removed = true;
                    for rule in env.get_rules().iter() {
                        let mut query = Query::new([*term.clone()]);
                        // let is_reducible = query.has_vars();
                        // if is_reducible {
                        //     debug!("{} still has free variables, so it is reducible", goal);
                        //     result.insert(goal.clone());
                        //     removed = false;
                        //     break;
                        // }

                        let mut tmp_env = env.clone();
    
                        if rule.apply(term, &mut query, &mut tmp_env) {
                            debug!("Rule {} applies to {}", rule, term);
                            removed = false;
                            result.insert(goal.clone());
                            break;
                        } else {
                            debug!("Rule {} does not apply to {}", rule, term);
                        }
                    }

                    if removed {
                        debug!("Removed irreducible negative goal with no possible branches: {}", goal);
                    }
                }
                _ => {
                    result.insert(goal.clone());
                }
            }
        }
        Query::new(result).reduce(env)
    }

    /// Is the query empty, i.e. does it have no goals left to prove?
    pub fn is_empty(&self) -> bool {
        self.goals.is_empty()
    }

    /// Is this query a ground truth query, meaning it is a query that is always true?
    /// 
    /// A query is ground truth if all the goals are either true or the complement of false.
    /// An empty query is also considered ground truth.
    pub fn is_ground_truth(&self) -> bool {
        if self.is_empty() {
            return true;
        }
    
        for goal in self.goals.iter() {
            // Each term must either be true, or complement of false
            match goal {
                Term::Complement(term) if term.as_ref() == &Term::False => {}
                Term::True => {}
                _ => {
                    return false;
                }
            }
        }
        
        true
    }

    /// Get an iterator over all the goals in the query
    pub fn goals(&self) -> impl Iterator<Item = &Term> {
        self.goals.iter()
    }

    /// Remove a goal from the query
    pub fn remove_goal(&mut self, goal: &Term) {
        // self.goals.remove(goal);
        Arc::make_mut(&mut self.goals).remove(goal);
    }

    /// Add a positive goal to the query. This means that the goal must be
    /// proven true for the query to be true.
    pub fn add_positive_goal(&mut self, goal: Term) {
        if goal == Term::True {
            return;
        }
        if let Term::Complement(term) = goal {
            self.add_negative_goal(*term);
            return;
        }
        // self.goals.insert(goal);
        Arc::make_mut(&mut self.goals).insert(goal);
    }

    /// Add a negative goal to the query. This means that the goal must be
    /// proven false for the query to be true.
    pub fn add_negative_goal(&mut self, goal: Term) {
        if goal == Term::False {
            return;
        }

        // self.goals.insert(goal.negate());
        Arc::make_mut(&mut self.goals).insert(goal.negate());
    }

    /// Does this query contain a goal that contradicts the given term?
    pub fn does_contradict_goals(&self, other_goal: &Term) -> bool {
        if other_goal == &Term::False {
            return true;
        }

        self.goals.iter().any(|goal| match goal {
            Term::Complement(term) => term.as_ref() == other_goal,
            _ => false,
        })
    }

    /// Does this query contain contradictory goals?
    /// 
    /// If false is in the query, then the query is contradictory.
    /// If a goal is in the query, and its complement is also in the query,
    /// then the query is contradictory.
    pub fn contains_contradiction(&self) -> bool {
        self.goals.iter().any(|goal| match goal {
            Term::Complement(term) => self.goals.contains(term),
            Term::False => true,
            _ => false,
        })
    }

    /// Get all the used variables in the query
    pub fn used_vars(&self, vars: &mut HashSet<Var>) {
        for goal in self.goals.iter() {
            goal.used_vars(vars);
        }
    }

    /// Does the query contain any variables?
    pub fn has_vars(&self) -> bool {
        for goal in self.goals.iter() {
            if goal.has_vars() {
                return true;
            }
        }
        false
    }

    /// Substitute all occurrences of some variables in the query with the given terms
    pub fn substitute(&mut self, bindings: &HashMap<Var, Term>) {
        self.goals = Arc::new(self.goals.iter().map(|goal| {
            let mut goal = goal.clone();
            goal.substitute(bindings);
            goal
        }).collect());
    }

    /// Simplify the query by substituting an environment's variable bindings
    /// into the query's goals. This makes the query's goals more concrete.
    /// 
    /// This does not mutate the query, but returns a new query with the
    /// substituted goals.
    pub fn reduce<S>(&self, env: &Env<S>) -> Self where S: Solver {
        Self::new(self.goals.iter().map(|goal| goal.reduce(env)))
    }

    /// Simplify the query by substituting an environment's variable bindings
    /// into the query's goals. This makes the query's goals more concrete.
    /// 
    /// This mutates the query in place.
    pub fn reduce_in_place(&mut self, env: &Env<impl Solver>) {
        self.goals = Arc::new(self.goals.iter().map(|goal| goal.reduce(env)).collect());
    }
}

/// Parse a query from a string
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

impl Display for Query {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "?- ")?;
        for (i, goal) in self.goals.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", goal)?;
        }
        write!(f, ".")
    }
}

impl Index<usize> for Query {
    type Output = Term;

    fn index(&self, index: usize) -> &Self::Output {
        self.goals.iter().nth(index).unwrap()
    }
}