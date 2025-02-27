use ordered_float::OrderedFloat;
use std::hash::{Hash, Hasher};

pub mod index;
// pub mod layer;

#[derive(Debug, Clone, Copy)]
pub struct DistEntry<T>(pub OrderedFloat<f32>, pub T);

impl<T: Eq> Eq for DistEntry<T> {}
impl<T: PartialEq> PartialEq for DistEntry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}
impl<T: PartialEq> PartialOrd for DistEntry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.0.cmp(&other.0))
    }
}
impl<T: Eq> Ord for DistEntry<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl<T: Hash> Hash for DistEntry<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.1.hash(state);
    }
}
