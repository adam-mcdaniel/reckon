use super::*;
use std::collections::{BTreeSet, HashMap};
use std::ops::Index;
use std::str::FromStr;
use std::sync::Arc;

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Query {
    goals: Arc<BTreeSet<Term>>,
}

impl Query {
    pub fn new(goals: impl IntoIterator<Item = Term>) -> Self {
        Query { goals: Arc::new(goals.into_iter().collect()) }
    }

    pub fn empty() -> Self {
        Query::new(vec![])
    }

    pub fn size(&self) -> usize {
        self.goals.iter().map(|goal| goal.size()).sum()
    }

    pub fn remove_irreducible_negatives_in_place(&mut self, env: &Env<impl Solver>) {
        if !self.has_complements() {
            return;
        }
        self.goals = self.remove_irreducible_negatives(env).goals;
    }

    pub fn has_complements(&self) -> bool {
        self.goals.iter().any(|goal| match goal {
            Term::Complement(_) => true,
            _ => false,
        })
    }

    pub fn has_only_complements(&self) -> bool {
        self.goals.iter().all(|goal| match goal {
            Term::Complement(_) => true,
            _ => false,
        })
    }

    pub fn filter_out_complements(&self) -> Self {
        Query::new(self.goals.iter().filter(|goal| match goal {
            Term::Complement(_) => false,
            _ => true,
        }).cloned())
    }

    pub fn join(&self, other: &Query) -> Self {
        Query::new(self.goals.iter().chain(other.goals.iter()).cloned())
    }

    // Returns true if absurdity was found
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
                    for rule in env.rules().iter() {
                        let mut query = Query::new([*term.clone()]);
                        // let is_reducible = query.has_used_vars();
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

    pub fn is_ground_truth(&self) -> bool {
        if self.goals.is_empty() {
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

    pub fn goals(&self) -> impl Iterator<Item = &Term> {
        self.goals.iter()
    }

    pub fn remove_goal(&mut self, goal: &Term) {
        // self.goals.remove(goal);
        Arc::make_mut(&mut self.goals).remove(goal);
    }

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

    pub fn add_negative_goal(&mut self, goal: Term) {
        if goal == Term::False {
            return;
        }

        // self.goals.insert(goal.negate());
        Arc::make_mut(&mut self.goals).insert(goal.negate());
    }

    pub fn does_contradict_goals(&self, other_goal: &Term) -> bool {
        if other_goal == &Term::False {
            return true;
        }

        self.goals.iter().any(|goal| match goal {
            Term::Complement(term) => term.as_ref() == other_goal,
            _ => false,
        })
    }

    pub fn contains_contradiction(&self) -> bool {
        self.goals.iter().any(|goal| match goal {
            Term::Complement(term) => self.goals.contains(term),
            Term::False => true,
            _ => false,
        })
    }

    pub fn used_vars(&self, vars: &mut HashSet<Var>) {
        for goal in self.goals.iter() {
            goal.used_vars(vars);
        }
    }

    pub fn has_used_vars(&self) -> bool {
        for goal in self.goals.iter() {
            if goal.has_used_vars() {
                return true;
            }
        }
        false
    }

    pub fn substitute(&mut self, bindings: &HashMap<Var, Term>) {
        self.goals = Arc::new(self.goals.iter().map(|goal| {
            let mut goal = goal.clone();
            goal.substitute(bindings);
            goal
        }).collect());
    }

    pub fn reduce<S>(&self, env: &Env<S>) -> Self where S: Solver {
        Self::new(self.goals.iter().map(|goal| goal.reduce(env)))
    }

    pub fn reduce_in_place(&mut self, env: &Env<impl Solver>) {
        self.goals = Arc::new(self.goals.iter().map(|goal| goal.reduce(env)).collect());
    }
}

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