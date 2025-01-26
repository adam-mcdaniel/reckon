use std::{
    // Import the borrow trait for borrowing symbols as strings.
    borrow::Borrow,
    // Import BTreeMap and HashMap from the standard library.
    // BTreeMap is used for tree expressions, which are ordered maps.
    // HashMap is used for map expressions, which are unordered maps,
    // and for the symbol table plus environment bindings.
    collections::HashMap,
    // Import the necessary types and traits for formatting our output.
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    // Import hash types and traits for hashing our expressions,
    // to allow them to be used as keys in a hash map.
    hash::Hash,
    // Import the string parsing trait for parsing symbols from strings.
    str::FromStr,
    // Import atomic reference counting for shared ownership of symbols,
    // and read-write locks for the symbol table.
    sync::{Arc, RwLock},
};

// Use lazy_static for setting up the symbol table as a global variable.
use lazy_static::lazy_static;

///////////////////////////////////////////////////////////////
// SYMBOLS AND SYMBOL TABLE
///////////////////////////////////////////////////////////////

/*
 * The symbol table is a hash map that maps strings to symbols.
 * It uses string interning to ensure that symbols are unique,
 * and to allow for fast comparison of symbols.
 */

lazy_static! {
    /// The symbol table that maps strings to symbols
    ///
    /// This is a global variable that is shared between all environments.
    /// It is a read-write lock that allows for multiple environments to
    /// read from the symbol table at the same time, but only one environment
    /// to write to the symbol table at a time.
    static ref SYMBOLS: RwLock<HashMap<String, Symbol>> = RwLock::new(HashMap::new());
    static ref SYMBOL_ID: RwLock<HashMap<Symbol, u64>> = RwLock::new(HashMap::new());
    static ref ID_SYMBOL: RwLock<HashMap<u64, Symbol>> = RwLock::new(HashMap::new());
    static ref ID_COUNTER: RwLock<u64> = RwLock::new(0);
}

fn new_symbol(name: &str) -> (Symbol, u64) {
    // First, check if the symbol already exists in the symbol table
    let mut symbols = SYMBOLS.write().unwrap();
    let mut symbol_to_id = SYMBOL_ID.write().unwrap();
    let mut id_to_symbol = ID_SYMBOL.write().unwrap();
    if let Some(symbol) = symbols.get(name) {
        let id = symbol_to_id.get(symbol).unwrap();
        return (symbol.clone(), *id);
    }

    // Otherwise, create a new symbol and add it to the symbol table
    let symbol = Symbol(Arc::new(name.to_string()));
    symbols.insert(name.to_string(), symbol.clone());

    // Create a new ID for the symbol
    let mut id_counter = ID_COUNTER.write().unwrap();
    *id_counter += 1;
    let id = *id_counter;

    symbol_to_id.insert(symbol.clone(), id);
    id_to_symbol.insert(id, symbol.clone());

    (symbol, id)
}

/// A symbol that uses string interning
#[derive(Clone, Hash, Eq, Ord)]
pub struct Symbol(Arc<String>);

impl Symbol {
    /// Create a new symbol from a string
    ///
    /// If the symbol already exists in the symbol table, it will return the existing symbol.
    /// Otherwise, it will create a new symbol and add it to the symbol table.
    pub fn new(name: &str) -> Self {
        new_symbol(name).0
    }

    pub fn already_exists(name: &str) -> bool {
        let symbols = SYMBOLS.read().unwrap();
        symbols.contains_key(name)
    }

    pub fn id(&self) -> u64 {
        let name = self.0.clone();
        let ids = SYMBOL_ID.read().unwrap();
        let name_str: &str = name.as_str();
        *ids.get(name_str).unwrap()
    }

    pub fn unused_id() -> u64 {
        let mut id_counter = ID_COUNTER.write().unwrap();
        *id_counter += 1;
        let id = *id_counter;
        id
    }

    pub fn from_id(id: u64) -> Self {
        let id_to_symbol = ID_SYMBOL.read().unwrap();
        if let Some(symbol) = id_to_symbol.get(&id) {
            return symbol.clone();
        }
        panic!("Symbol with ID {} does not exist", id);
    }

    /// Get the name of the symbol as a string
    ///
    /// This is useful when you need the internal string representation of the symbol.
    pub fn name(&self) -> &str {
        &self.0
    }

    /// Borrow the name of the symbol as a string
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }

    pub fn refresh(&self) -> Self {
        lazy_static! {
            static ref FRESH_COUNTER: RwLock<u64> = RwLock::new(0);
        }
        let mut counter = FRESH_COUNTER.write().unwrap();
        let id = *counter;
        *counter += 1;
        new_symbol(&format!("{}-refreshed-#{}", self, id)).0
    }
}

/// Convert a &str to a symbol conveniently
///
/// This allows you to pass a string to a function that expects a symbol,
/// using the `into()` method.
impl From<&str> for Symbol {
    #[inline]
    fn from(s: &str) -> Self {
        Symbol::new(s)
    }
}

/// Convert a String to a symbol conveniently
///
/// This allows you to pass a string to a function that expects a symbol,
/// using the `into()` method.
impl From<String> for Symbol {
    #[inline]
    fn from(s: String) -> Self {
        Symbol::new(&s)
    }
}

/// Parse a symbol from a string
impl FromStr for Symbol {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Symbol::new(s))
    }
}

/// Compare two symbols for equality
///
/// This allows you to compare two symbols using the `==` operator.
/// First, it checks if the two symbols are the same object in memory.
/// If they are not, it compares the internal strings of the symbols.
///
/// This is faster than comparing the strings directly, because a pointer comparison
/// is faster than a string comparison.
impl PartialEq for Symbol {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // Check if the two symbols are the same object in memory
        if Arc::ptr_eq(&self.0, &other.0) {
            return true;
        }
        // Compare the internal strings of the symbols
        self.0 == other.0
    }
}

/// Compare two symbols for ordering.
///
/// If the two symbols are the same object in memory, they are equal.
/// Otherwise, it compares the internal strings of the symbols.
impl PartialOrd for Symbol {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if Arc::ptr_eq(&self.0, &other.0) {
            return Some(std::cmp::Ordering::Equal);
        }
        self.0.partial_cmp(&other.0)
    }
}

/// Print a symbol as standard output
impl Display for Symbol {
    #[inline]
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{}", self.0)
    }
}

/// Print a symbol as debug output
///
/// Since a symbol is meant to be an identifier, it is printed as a normal string.
/// This is useful for debugging, because it allows you to distinguish symbols from strings.
impl Debug for Symbol {
    #[inline]
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{}", self.0)
    }
}

impl Borrow<str> for Symbol {
    fn borrow(&self) -> &str {
        self.0.as_str()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol() {
        let symbol1 = Symbol::new("foo");
        let symbol2 = Symbol::new("foo");
        let symbol3 = Symbol::new("bar");

        assert_eq!(symbol1, symbol2);
        assert_ne!(symbol1, symbol3);
        assert_ne!(symbol2, symbol3);
    }

    // Test the ID generation for symbols
    #[test]
    fn test_symbol_id() {
        let symbol1 = Symbol::new("foo");
        let symbol2 = Symbol::new("foo");
        let symbol3 = Symbol::new("bar");

        assert_eq!(symbol1.id(), symbol2.id());
        assert_ne!(symbol1.id(), symbol3.id());
        assert_ne!(symbol2.id(), symbol3.id());

        // Assert the later IDs are greater than the earlier IDs
        assert!(symbol1.id() < symbol3.id());
        assert!(symbol2.id() < symbol3.id());
    }
}
