//! Phase 105 — Family tree / blood relations (Promot: "family tree")
//! (Rust port of python/puzzles/family_tree_domain.py).
//!
//! Ground truth: build an explicit directed relation graph from stated
//! facts ("A is the father of B" etc.), then answer "what is A to B?" by
//! walking the graph and matching the resulting parent/spouse-chain
//! pattern against a table of known relationship names -- derive from an
//! explicit structure rather than guessing.
//!
//! Supported base facts: father, mother, son, daughter, husband, wife,
//! brother, sister (siblings share both parents here -- no half-sibling
//! modeling, stated as a scope limit).

use std::collections::{HashMap, HashSet};

/// `FamilyTree` dataclass with `field(default_factory=dict)` fields.
#[derive(Debug, Clone, Default)]
pub struct FamilyTree {
    /// person -> set of parents
    pub parents: HashMap<String, HashSet<String>>,
    /// person -> spouse
    pub spouses: HashMap<String, String>,
    /// person -> 'M'/'F' (when statable)
    pub genders: HashMap<String, String>,
}

impl FamilyTree {
    pub fn new() -> FamilyTree {
        FamilyTree::default()
    }

    /// 'subject is the <relation> of obj', e.g. add_fact('A','father','B').
    pub fn add_fact(&mut self, subject: &str, relation: &str, obj: &str) -> Result<(), String> {
        let relation = relation.to_lowercase();
        match relation.as_str() {
            "father" => {
                self.parents
                    .entry(obj.to_string())
                    .or_default()
                    .insert(subject.to_string());
                self.genders.insert(subject.to_string(), "M".to_string());
            }
            "mother" => {
                self.parents
                    .entry(obj.to_string())
                    .or_default()
                    .insert(subject.to_string());
                self.genders.insert(subject.to_string(), "F".to_string());
            }
            "son" => {
                self.parents
                    .entry(subject.to_string())
                    .or_default()
                    .insert(obj.to_string());
                self.genders.insert(subject.to_string(), "M".to_string());
            }
            "daughter" => {
                self.parents
                    .entry(subject.to_string())
                    .or_default()
                    .insert(obj.to_string());
                self.genders.insert(subject.to_string(), "F".to_string());
            }
            "husband" => {
                self.spouses.insert(subject.to_string(), obj.to_string());
                self.spouses.insert(obj.to_string(), subject.to_string());
                self.genders.insert(subject.to_string(), "M".to_string());
                self.genders.insert(obj.to_string(), "F".to_string());
            }
            "wife" => {
                self.spouses.insert(subject.to_string(), obj.to_string());
                self.spouses.insert(obj.to_string(), subject.to_string());
                self.genders.insert(subject.to_string(), "F".to_string());
                self.genders.insert(obj.to_string(), "M".to_string());
            }
            "brother" | "sister" => {
                // shares both parents with obj, if obj's parents are already
                // known (clone first: the entry() below needs &mut self)
                if let Some(obj_parents) = self.parents.get(obj) {
                    let shared: Vec<String> = obj_parents.iter().cloned().collect();
                    self.parents
                        .entry(subject.to_string())
                        .or_default()
                        .extend(shared);
                }
                self.genders.insert(
                    subject.to_string(),
                    if relation == "brother" {
                        "M".to_string()
                    } else {
                        "F".to_string()
                    },
                );
            }
            _ => {
                return Err(format!("unsupported relation: '{}'", relation));
            }
        }
        Ok(())
    }

    /// BFS up the parent graph -- distance in generations.
    /// (Python default `max_gen: int = 4`.)
    fn _ancestors_with_distance(&self, person: &str, max_gen: i32) -> HashMap<String, i32> {
        let mut dist: HashMap<String, i32> = HashMap::new();
        dist.insert(person.to_string(), 0);
        let mut frontier: Vec<String> = vec![person.to_string()];
        let mut gen = 0;
        while !frontier.is_empty() && gen < max_gen {
            gen += 1;
            let mut nxt: Vec<String> = Vec::new();
            for p in &frontier {
                if let Some(parents) = self.parents.get(p) {
                    for parent in parents {
                        if !dist.contains_key(parent) {
                            dist.insert(parent.clone(), gen);
                            nxt.push(parent.clone());
                        }
                    }
                }
            }
            frontier = nxt;
        }
        dist
    }

    /// What is `a` to `b`? (e.g. relationship('A','B') = "father" means
    /// A is B's father.) Returns None if not derivable from known facts
    /// rather than guessing.
    pub fn relationship(&self, a: &str, b: &str) -> Option<String> {
        if a == b {
            return None;
        }

        if self.spouses.get(a).map(|s| s.as_str()) == Some(b) {
            return match self.genders.get(a).map(|s| s.as_str()) {
                Some("M") => Some("husband".to_string()),
                Some("F") => Some("wife".to_string()),
                _ => Some("spouse".to_string()),
            };
        }

        let anc_a = self._ancestors_with_distance(a, 4);
        let anc_b = self._ancestors_with_distance(b, 4);

        // direct ancestor of b?
        if let Some(&gen) = anc_b.get(a) {
            return Some(_ancestor_label(gen, self.genders.get(a).map(|s| s.as_str())));
        }
        if let Some(&gen) = anc_a.get(b) {
            return Some(_descendant_label(gen, self.genders.get(a).map(|s| s.as_str())));
        }

        // common ancestor -> sibling / cousin / aunt-uncle / niece-nephew
        let mut common: Vec<&String> = anc_a
            .keys()
            .filter(|k| anc_b.contains_key(*k))
            .collect();
        common.retain(|c| c.as_str() != a && c.as_str() != b);
        if common.is_empty() {
            return None;
        }
        // closest common ancestor (Python `min` over a set has arbitrary
        // tie order; we tie-break deterministically by name)
        common.sort();
        let closest = common
            .into_iter()
            .min_by_key(|c| anc_a[*c] + anc_b[*c])
            .unwrap();
        let ga = anc_a[closest];
        let gb = anc_b[closest];
        let gender_a = self.genders.get(a).map(|s| s.as_str());
        if ga == 1 && gb == 1 {
            return Some(
                match gender_a {
                    Some("M") => "brother",
                    Some("F") => "sister",
                    _ => "sibling",
                }
                .to_string(),
            );
        }
        if ga == 1 && gb == 2 {
            return Some(
                match gender_a {
                    Some("M") => "uncle",
                    Some("F") => "aunt",
                    _ => "aunt/uncle",
                }
                .to_string(),
            );
        }
        if ga == 2 && gb == 1 {
            return Some(
                match gender_a {
                    Some("M") => "nephew",
                    Some("F") => "niece",
                    _ => "nephew/niece",
                }
                .to_string(),
            );
        }
        if ga == 2 && gb == 2 {
            return Some("cousin".to_string());
        }
        None
    }
}

fn _ancestor_label(gen: i32, gender: Option<&str>) -> String {
    if gen == 1 {
        return match gender {
            Some("M") => "father",
            Some("F") => "mother",
            _ => "parent",
        }
        .to_string();
    }
    if gen == 2 {
        return match gender {
            Some("M") => "grandfather",
            Some("F") => "grandmother",
            _ => "grandparent",
        }
        .to_string();
    }
    format!("ancestor ({} generations up)", gen)
}

fn _descendant_label(gen: i32, gender: Option<&str>) -> String {
    if gen == 1 {
        return match gender {
            Some("M") => "son",
            Some("F") => "daughter",
            _ => "child",
        }
        .to_string();
    }
    if gen == 2 {
        return match gender {
            Some("M") => "grandson",
            Some("F") => "granddaughter",
            _ => "grandchild",
        }
        .to_string();
    }
    format!("descendant ({} generations down)", gen)
}
