use std::{
    cmp::Ordering,
    hash::Hash,
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use arrayvec::ArrayVec;
use ordered_float::OrderedFloat;

use crate::DistEntry;

pub trait Heap<T> {
    fn push(&mut self, val: DistEntry<T>) -> Option<DistEntry<T>>;
}

pub trait DistanceCache<T> {
    fn get(&self, key: (T, T)) -> Option<f32>;
    fn put(&self, key: (T, T), dist: f32);
}

pub type LeapMapCache<K> = leapfrog::LeapMap<(K, K), Wrap>;

impl<K: Eq + Hash + Copy + PartialOrd> DistanceCache<K> for LeapMapCache<K> {
    #[inline]
    fn get(&self, key: (K, K)) -> Option<f32> {
        let key = if key.0 <= key.1 {
            (key.0, key.1)
        } else {
            (key.1, key.0)
        };

        self.get(&key)?.value().map(|Wrap(inner)| inner)
    }

    #[inline]
    fn put(&self, key: (K, K), dist: f32) {
        let key = if key.0 <= key.1 {
            (key.0, key.1)
        } else {
            (key.1, key.0)
        };

        if !self.contains_key(&key) {
            self.insert(key, Wrap(dist));
        }
    }
}

impl<K: Eq + Hash + Copy, U: DistanceCache<K>> DistanceCache<K> for &U {
    #[inline]
    fn get(&self, key: (K, K)) -> Option<f32> {
        (**self).get(key)
    }

    #[inline]
    fn put(&self, key: (K, K), dist: f32) {
        (**self).put(key, dist)
    }
}

pub trait Collection<T>: Default + Deref<Target = [T]> + DerefMut {
    type Type<Q>: Collection<Q>;

    fn push(&mut self, item: T) -> usize;
    fn pop(&mut self) -> Option<T>;
}

impl<T> Collection<T> for Vec<T> {
    type Type<Q> = Vec<Q>;

    #[inline]
    fn push(&mut self, item: T) -> usize {
        let index = self.len();
        self.push(item);
        index
    }

    #[inline]
    fn pop(&mut self) -> Option<T> {
        self.pop()
    }
}
impl<T, const N: usize> Collection<T> for ArrayVec<T, N> {
    type Type<Q> = ArrayVec<Q, N>;

    #[inline]
    fn push(&mut self, item: T) -> usize {
        let index = self.len();
        self.push(item);
        index
    }

    #[inline]
    fn pop(&mut self) -> Option<T> {
        self.pop()
    }
}

#[derive(Debug)]
pub struct DistanceOrderedHeap<'a, T, A>
where
    T: Ord + Eq + Copy,
    A: Collection<T>,
{
    inner: &'a mut A,
    dists: A::Type<OrderedFloat<f32>>,
    k_limit: u32,
    dist_limit: f32,
}

impl<T, A> Heap<T> for DistanceOrderedHeap<'_, T, A>
where
    T: Ord + Eq + Copy,
    A: Collection<T>,
{
    fn push(&mut self, val: DistEntry<T>) -> Option<DistEntry<T>> {
        self.push(val)
    }
}

impl<'a, T, A> DistanceOrderedHeap<'a, T, A>
where
    T: Ord + Eq + Copy,
    A: Collection<T>,
{
    pub fn new(inner: &'a mut A, k_limit: u32, dist_limit: f32) -> Self {
        Self {
            inner,
            dists: Default::default(),
            k_limit,
            dist_limit,
        }
    }
    pub fn with_cache<C, D>(
        inner: &'a mut A,
        node: T,
        cache: C,
        distance: D,
        max_k: Option<u32>,
        max_dist: Option<f32>,
    ) -> Self
    where
        C: DistanceCache<T>,
        D: Fn(T, T) -> f32,
    {
        let mut dists: A::Type<OrderedFloat<f32>> = Default::default();

        for item in inner.iter() {
            let dist = if let Some(dist) = cache.get((node, *item)) {
                dist
            } else {
                (distance)(node, *item)
            };

            dists.push(OrderedFloat(dist));
        }

        Self {
            inner,
            dists,
            k_limit: max_k.unwrap_or(u32::MAX),
            dist_limit: max_dist.unwrap_or(f32::INFINITY),
        }
    }

    pub fn push(&mut self, entry: DistEntry<T>) -> Option<DistEntry<T>> {
        if self.inner.contains(&entry.1) {
            return None;
        }

        if entry.0 > OrderedFloat(self.dist_limit) {
            return Some(entry);
        }

        if self.dists.len() >= self.k_limit as usize {
            if self.dists[0] > entry.0 {
                let tmp = self.pop();
                self.do_push(entry.1, entry.0);
                tmp
            } else {
                Some(entry)
            }
        } else {
            self.do_push(entry.1, entry.0);
            None
        }
    }

    fn do_push(&mut self, item: T, dist: OrderedFloat<f32>) {
        self.inner.push(item);
        self.dists.push(dist);

        let n = self.inner.len();

        let mut i = n;
        let mut p = i / 2;

        while p > 0 && (self.dists[i - 1].cmp(&self.dists[p - 1]) == Ordering::Greater) {
            self.dists.swap(p - 1, i - 1);
            self.inner.swap(p - 1, i - 1);

            i = p;
            p = i / 2;
        }
    }

    pub fn pop(&mut self) -> Option<DistEntry<T>> {
        let n = self.inner.len();

        if n > 1 {
            self.inner.swap(0, n - 1);
            self.dists.swap(0, n - 1);

            let n = self.inner.len() - 1;
            let mut i = 0;
            let mut l = i * 2 + 1;
            let mut r = i * 2 + 2;

            if r < n && self.dists[r].cmp(&self.dists[l]) == Ordering::Greater {
                l = r;
            }

            // invariant: slice[l] >= slice[r]
            // if slice[l] > slice[i], do push
            while l < n && self.dists[l].cmp(&self.dists[i]) == Ordering::Greater {
                self.dists.swap(i, l);
                self.inner.swap(i, l);

                i = l;
                l = i * 2 + 1;
                r = i * 2 + 2;
                if r < n && self.dists[r].cmp(&self.dists[l]) == Ordering::Greater {
                    l = r;
                }
            }

            // bubble_down(&mut slice[0..n - 1], 0, pred);
        }

        let item = self.inner.pop()?;

        Some(DistEntry(self.dists.pop().unwrap(), item))
    }

    #[inline]
    pub(crate) fn iter(&self) -> impl Iterator<Item = (T, f32)> + '_ {
        std::iter::zip(
            self.inner.iter().copied(),
            self.dists.iter().map(|OrderedFloat(f)| *f),
        )
    }
}

// fn bubble_up<T, P: Fn(&T, &T) -> Ordering>(slice: &mut [T], pred: P) {
//     let n = slice.len();
//     let mut i = n;
//     let mut p = i / 2;
//     while p > 0 && (pred_greater(&pred, &slice[i - 1], &slice[p - 1])) {
//         slice.swap(p - 1, i - 1);
//         i = p;
//         p = i / 2;
//     }
// }

// fn bubble_down<T, P: Fn(&T, &T) -> Ordering>(slice: &mut [T], index: usize, pred: P) {
//     let n = slice.len();
//     let mut i = index;
//     let mut l = i * 2 + 1;
//     let mut r = i * 2 + 2;

//     if r < n && pred_greater(&pred, &slice[r], &slice[l]) {
//         l = r;
//     }

//     // invariant: slice[l] >= slice[r]
//     // if slice[l] > slice[i], do push
//     while l < n && pred_greater(&pred, &slice[l], &slice[i]) {
//         slice.swap(i, l);

//         i = l;
//         l = i * 2 + 1;
//         r = i * 2 + 2;
//         if r < n && pred_greater(&pred, &slice[r], &slice[l]) {
//             l = r;
//         }
//     }
// }

#[derive(Debug, Clone, Copy)]
pub struct Wrap(pub f32);

impl Deref for Wrap {
    type Target = f32;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl PartialEq for Wrap {
    fn eq(&self, other: &Self) -> bool {
        OrderedFloat(self.0) == OrderedFloat(other.0)
    }
}

impl Eq for Wrap {}

impl PartialOrd for Wrap {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Wrap {
    fn cmp(&self, other: &Self) -> Ordering {
        OrderedFloat(self.0).cmp(&OrderedFloat(other.0))
    }
}

impl leapfrog::Value for Wrap {
    fn is_redirect(&self) -> bool {
        self.0 == f32::INFINITY
    }

    fn is_null(&self) -> bool {
        self.0.is_nan()
    }

    fn redirect() -> Self {
        Wrap(f32::INFINITY)
    }

    fn null() -> Self {
        Wrap(f32::NAN)
    }
}

pub struct MappedHeap<'a, T, H, F> {
    inner: &'a mut H,
    map: F,
    _m: PhantomData<T>,
}
impl<'a, T, H, F> MappedHeap<'a, T, H, F> {
    pub(crate) fn new(inner: &'a mut H, map: F) -> Self {
        Self {
            inner,
            map,
            _m: PhantomData,
        }
    }
}

impl<T, U: Copy, H, F> Heap<U> for MappedHeap<'_, T, H, F>
where
    H: Heap<T>,
    F: Fn(U) -> T,
{
    fn push(&mut self, val: DistEntry<U>) -> Option<DistEntry<U>> {
        self.inner
            .push(DistEntry(val.0, (self.map)(val.1)))
            .map(|v| DistEntry(v.0, val.1))
    }
}

#[cfg(test)]
mod tests {
    use arrayvec::ArrayVec;
    use ordered_float::OrderedFloat;

    use crate::DistEntry;

    use super::{DistanceOrderedHeap, LeapMapCache};

    #[test]
    fn test_heap_arrayvec() {
        let cache = LeapMapCache::new();
        let mut data: ArrayVec<u32, 8> = ArrayVec::new();
        {
            let mut wrapper = DistanceOrderedHeap::with_cache(
                &mut data,
                12,
                &cache,
                |_, b| (b as f32 - 12.0f32).abs(),
                Some(8),
                None,
            );

            assert_eq!(wrapper.push(DistEntry(OrderedFloat(11.0), 1)), None);
            assert_eq!(wrapper.push(DistEntry(OrderedFloat(7.0), 5)), None);
            assert_eq!(wrapper.push(DistEntry(OrderedFloat(5.0), 7)), None);
            assert_eq!(wrapper.push(DistEntry(OrderedFloat(3.0), 9)), None);
            assert_eq!(wrapper.push(DistEntry(OrderedFloat(4.0), 16)), None);
            assert_eq!(wrapper.push(DistEntry(OrderedFloat(1.0), 11)), None);
            assert_eq!(wrapper.push(DistEntry(OrderedFloat(13.0), 25)), None);
            assert_eq!(wrapper.push(DistEntry(OrderedFloat(2.0), 14)), None);
            assert_eq!(
                wrapper.push(DistEntry(OrderedFloat(8.0), 20)),
                Some(DistEntry(OrderedFloat(13.0), 25))
            );
            assert_eq!(wrapper.pop(), Some(DistEntry(OrderedFloat(11.0), 1)));
            assert_eq!(wrapper.pop(), Some(DistEntry(OrderedFloat(8.0), 20)));
            assert_eq!(wrapper.pop(), Some(DistEntry(OrderedFloat(7.0), 5)));
            assert_eq!(wrapper.pop(), Some(DistEntry(OrderedFloat(5.0), 7)));
            assert_eq!(wrapper.pop(), Some(DistEntry(OrderedFloat(4.0), 16)));
            assert_eq!(wrapper.pop(), Some(DistEntry(OrderedFloat(3.0), 9)));
            assert_eq!(wrapper.pop(), Some(DistEntry(OrderedFloat(2.0), 14)));
            assert_eq!(wrapper.pop(), Some(DistEntry(OrderedFloat(1.0), 11)));
            assert_eq!(wrapper.pop(), None);
        }
    }
}
