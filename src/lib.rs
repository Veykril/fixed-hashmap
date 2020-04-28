#![cfg_attr(not(test), no_std)]
#![feature(
    const_generics,
    maybe_uninit_uninit_array,
    maybe_uninit_slice_assume_init,
    maybe_uninit_ref
)]

use core::borrow::Borrow;
use core::fmt;
use core::hash::{BuildHasher, Hash, Hasher};
use core::iter::{ExactSizeIterator, FusedIterator};
use core::mem::MaybeUninit;
use core::ptr;
use core::slice;

pub struct HashMap<K, V, S, const LEN: usize> {
    hash_builder: S,
    table: Table<K, V, LEN>,
}

impl<K: fmt::Debug, V: fmt::Debug, S, const LEN: usize> fmt::Debug for HashMap<K, V, S, LEN> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HashMap")
            .field("table", &self.table)
            .finish()
    }
}

struct Table<K, V, const LEN: usize>([Node<K, V>; LEN]);

impl<K, V, const LEN: usize> Table<K, V, LEN>
where
    K: Hash,
{
    pub fn new<I>(hash_builder: &impl BuildHasher, iter: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        I::IntoIter: ExactSizeIterator,
    {
        let iter = iter.into_iter();
        if iter.len() != LEN {
            unimplemented!("err")
        }

        // TODO: drop initialized elements on panic. even if we don't panic anywhere, the iterator might
        let mut array: [MaybeUninit<Node<K, V>>; LEN] = MaybeUninit::uninit_array();
        let mut slot_map: [bool; LEN] = [false; LEN];

        for (key, value) in iter {
            let slot = hash_key(hash_builder, &key) as usize % LEN;
            // already occupied
            if slot_map[slot] {
                let next_free_slot = (0..LEN)
                    .map(|i| i + slot)
                    .find(|&idx| !slot_map[idx % LEN])
                    .expect("bug: array is full");
                let (prev, next) = unsafe {
                    let slot = array[slot].get_ref();
                    (slot.prev, slot.next)
                };
                // make room for new node, put previous in a free spot
                array.swap(slot, next_free_slot);
                // then write new node into the uninitialized memory and fix links
                // A head, prepend this new node to the list
                if prev == slot {
                    array[slot] = MaybeUninit::new(Node {
                        key,
                        value,
                        // head, so point prev to self
                        prev: slot,
                        // point to previous head
                        next: next_free_slot,
                    });
                    // SAFETY: `next_free_slot` has been initialized this iteration,
                    //  `next` has been initialized in an earlier iteration
                    unsafe {
                        // slot was single, so turn it into a tail
                        if next == slot {
                            array[next_free_slot].get_mut().next = next_free_slot;
                        // otherwise fix links due to element move
                        } else {
                            array[next].get_mut().prev = next_free_slot;
                            array[next_free_slot].get_mut().prev = slot;
                        }
                    }
                // Not a head, this new node becomes a new list
                } else {
                    array[slot] = MaybeUninit::new(Node {
                        key,
                        value,
                        // we are a new chain, so point to self
                        prev: slot,
                        next: slot,
                    });
                    // SAFETY: `next_free_slot` has been initialized this iteration,
                    //  `next` and `prev` have been initialized in an earlier iteration
                    unsafe {
                        // place the new node into the now free slot
                        array[prev].get_mut().next = next_free_slot;
                        // slot was a tail
                        if next == slot {
                            // uphold tail's invariant of pointing to itself
                            array[next_free_slot].get_mut().next = next_free_slot;
                        // slot was in the middle of a list
                        } else {
                            // update invalidated back pointer of the previous second element
                            array[next].get_mut().prev = next_free_slot;
                        }
                    }
                }

                slot_map[next_free_slot] = true;
            // free spot yay
            } else {
                array[slot] = MaybeUninit::new(Node {
                    key,
                    value,
                    // is head
                    prev: slot,
                    // is tail
                    next: slot,
                });
                slot_map[slot] = true;
            }
        }
        debug_assert!(slot_map.iter().copied().all(core::convert::identity));
        // SAFETY: the array has been properly initialized
        Table(unsafe { ptr::read(&mut array as *mut _ as *mut _) })
    }

    fn find(&self, hash: u64, predicate: impl Fn(&K) -> bool) -> Option<&Node<K, V>> {
        let mut last = (hash % LEN as u64) as usize;
        let mut entry = &self.0[last];
        loop {
            if predicate(&entry.key) {
                break Some(entry);
            }
            if entry.next != last {
                last = entry.next;
                entry = &self.0[entry.next];
            } else {
                break None;
            }
        }
    }

    fn find_mut(&mut self, hash: u64, predicate: impl Fn(&K) -> bool) -> Option<&mut Node<K, V>> {
        let mut entry_slot = (hash % LEN as u64) as usize;
        let mut entry = &self.0[entry_slot];
        loop {
            if predicate(&entry.key) {
                break Some(entry_slot);
            }
            if entry.next != entry_slot {
                entry_slot = entry.next;
                entry = &self.0[entry.next];
            } else {
                break None;
            }
        }
        .map(move |slot| &mut self.0[slot])
    }
}

impl<K: fmt::Debug, V: fmt::Debug, const LEN: usize> fmt::Debug for Table<K, V, LEN> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Table").field(&&self.0[..]).finish()
    }
}

#[derive(Debug, PartialEq)]
struct Node<K, V> {
    key: K,
    value: V,
    // if pref == self -> head
    // if pref == self && next == self -> single
    prev: usize,
    // if next = self -> tail
    next: usize,
}

fn hash_key<K: Hash + ?Sized>(hash_builder: &impl BuildHasher, key: &K) -> u64 {
    let mut state = hash_builder.build_hasher();
    key.hash(&mut state);
    state.finish()
}

impl<K, V, S, const LEN: usize> HashMap<K, V, S, LEN>
where
    K: Hash,
    S: BuildHasher + Default,
{
    pub fn new<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        I::IntoIter: ExactSizeIterator,
    {
        let hash_builder = S::default();
        let table = Table::new(&hash_builder, iter);
        HashMap {
            hash_builder,
            table,
        }
    }
}

impl<K, V, S, const LEN: usize> HashMap<K, V, S, LEN>
where
    K: Hash,
    S: BuildHasher,
{
    pub fn with_hasher<I>(iter: I, hash_builder: S) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        I::IntoIter: ExactSizeIterator,
    {
        let table = Table::new(&hash_builder, iter);
        HashMap {
            hash_builder,
            table,
        }
    }

    pub fn hasher(&self) -> &S {
        &self.hash_builder
    }
}

impl<K, V, S, const LEN: usize> HashMap<K, V, S, LEN> {
    pub fn keys(&self) -> Keys<'_, K, V> {
        Keys { inner: self.iter() }
    }

    pub fn values(&self) -> Values<'_, K, V> {
        Values { inner: self.iter() }
    }

    pub fn values_mut(&mut self) -> ValuesMut<'_, K, V> {
        ValuesMut {
            inner: self.iter_mut(),
        }
    }

    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            inner: self.table.0.iter(),
        }
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut {
            inner: self.table.0.iter_mut(),
        }
    }
}

impl<K, V, S, const LEN: usize> HashMap<K, V, S, LEN>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    pub fn get<Q: ?Sized>(&self, k: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.get_key_value(k).map(|(_, v)| v)
    }

    pub fn get_mut<Q: ?Sized>(&mut self, k: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.get_key_value_mut(k).map(|(_, v)| v)
    }

    pub fn get_key_value<Q: ?Sized>(&self, k: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let hash = hash_key(&self.hash_builder, k);
        self.table
            .find(hash, |key| key.borrow() == k)
            .map(|node| (&node.key, &node.value))
    }

    pub fn get_key_value_mut<Q: ?Sized>(&mut self, k: &Q) -> Option<(&K, &mut V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let hash = hash_key(&self.hash_builder, k);
        self.table
            .find_mut(hash, |key| key.borrow() == k)
            .map(|node| (&node.key, &mut node.value))
    }

    pub fn contains_key<Q: ?Sized>(&self, k: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.get(k).is_some()
    }
}

impl<'a, K, V, S, const LEN: usize> IntoIterator for &'a HashMap<K, V, S, LEN> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Iter<'a, K, V> {
        self.iter()
    }
}

impl<'a, K, V, S, const LEN: usize> IntoIterator for &'a mut HashMap<K, V, S, LEN> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;

    fn into_iter(self) -> IterMut<'a, K, V> {
        self.iter_mut()
    }
}

impl<K, V, S, const LEN: usize> IntoIterator for HashMap<K, V, S, LEN> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V, LEN>;

    fn into_iter(self) -> IntoIter<K, V, LEN> {
        IntoIter { table: self.table }
    }
}

pub struct IntoIter<K, V, const LEN: usize> {
    table: Table<K, V, LEN>,
}

impl<K, V, const LEN: usize> Iterator for IntoIter<K, V, LEN> {
    type Item = (K, V);

    fn next(&mut self) -> Option<(K, V)> {
        todo!()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        todo!()
    }
}

impl<K, V, const LEN: usize> FusedIterator for IntoIter<K, V, LEN> {}
impl<K, V, const LEN: usize> ExactSizeIterator for IntoIter<K, V, LEN> {
    fn len(&self) -> usize {
        todo!()
    }
}

impl<K: fmt::Debug, V: fmt::Debug, const LEN: usize> fmt::Debug for IntoIter<K, V, LEN> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

pub struct Iter<'a, K, V> {
    inner: slice::Iter<'a, Node<K, V>>,
}

pub struct IterMut<'a, K, V> {
    inner: slice::IterMut<'a, Node<K, V>>,
}

pub struct Keys<'a, K, V> {
    inner: Iter<'a, K, V>,
}

pub struct Values<'a, K, V> {
    inner: Iter<'a, K, V>,
}

pub struct ValuesMut<'a, K, V> {
    inner: IterMut<'a, K, V>,
}

macro_rules! derive_iter {
    ($ty:ty:Item = $item_ty:ty; -> $closure:expr) => {
        impl<'a, K, V> Iterator for $ty {
            type Item = $item_ty;

            fn next(&mut self) -> Option<Self::Item> {
                self.inner.next().map($closure)
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                self.inner.size_hint()
            }
        }

        impl<'a, K, V> FusedIterator for $ty {}
        impl<'a, K, V> ExactSizeIterator for $ty {
            fn len(&self) -> usize {
                self.inner.len()
            }
        }
    };
}
derive_iter!(Iter<'a, K, V>:Item = (&'a K, &'a V);          -> |node| (&node.key, &node.value));
derive_iter!(IterMut<'a, K, V>:Item = (&'a K, &'a mut V);   -> |node| (&node.key, &mut node.value));
derive_iter!(Keys<'a, K, V>:Item = &'a K;                   -> |(k, _)| k);
derive_iter!(Values<'a, K, V>:Item = &'a V;                 -> |(_, v)| v);
derive_iter!(ValuesMut<'a, K, V>:Item = &'a mut V;          -> |(_, v)| v);

#[cfg(test)]
mod test {
    use super::Node;
    type HashBuilder = core::hash::BuildHasherDefault<nohash_hasher::NoHashHasher<u32>>;
    type HashMap<const LEN: usize> = super::HashMap<u32, u32, HashBuilder, LEN>;

    #[test]
    fn test_get() {
        let foo = HashMap::<5>::new(vec![(0, 0), (3, 3), (5, 5), (8, 8), (1, 1)]);

        assert_eq!(foo.get(&0), Some(&0));
        assert_eq!(foo.get(&1), Some(&1));
        assert_eq!(foo.get(&3), Some(&3));
        assert_eq!(foo.get(&5), Some(&5));
        assert_eq!(foo.get(&8), Some(&8));
    }

    #[test]
    fn test_equal_elements() {
        // FIXME should this error, or is this fine and considered a logical error?
        let foo = HashMap::<2>::new(vec![(0, 0), (0, 0)]);

        assert_eq!(
            foo.table.0,
            [
                Node {
                    key: 0,
                    value: 0,
                    prev: 0,
                    next: 1,
                },
                Node {
                    key: 0,
                    value: 0,
                    prev: 0,
                    next: 1,
                },
            ],
        );
    }

    #[test]
    fn test_single_col() {
        let foo = HashMap::<2>::new(vec![(0, 0), (2, 2)]);

        assert_eq!(
            foo.table.0,
            [
                Node {
                    key: 2,
                    value: 2,
                    prev: 0,
                    next: 1,
                },
                Node {
                    key: 0,
                    value: 0,
                    prev: 0,
                    next: 1,
                },
            ],
        );
    }

    #[test]
    fn test_tail_col() {
        let foo = HashMap::<3>::new(vec![(0, 1), (3, 0), (4, 2)]);

        assert_eq!(
            foo.table.0,
            [
                Node {
                    key: 3,
                    value: 0,
                    prev: 0,
                    next: 2,
                },
                Node {
                    key: 4,
                    value: 2,
                    prev: 1,
                    next: 1,
                },
                Node {
                    key: 0,
                    value: 1,
                    prev: 0,
                    next: 2,
                },
            ],
        );
    }

    #[test]
    fn test_head_col() {
        let foo = HashMap::<3>::new(vec![(0, 1), (3, 0), (6, 2)]);

        assert_eq!(
            foo.table.0,
            [
                Node {
                    key: 6,
                    value: 2,
                    prev: 0,
                    next: 2,
                },
                Node {
                    key: 0,
                    value: 1,
                    prev: 2,
                    next: 1,
                },
                Node {
                    key: 3,
                    value: 0,
                    prev: 0,
                    next: 1,
                },
            ],
        );
    }

    #[test]
    fn test_inbetween_col() {
        let foo = HashMap::<4>::new(vec![(0, 1), (4, 0), (8, 2), (2, 3)]);

        assert_eq!(
            foo.table.0,
            [
                Node {
                    key: 8,
                    value: 2,
                    prev: 0,
                    next: 3,
                },
                Node {
                    key: 0,
                    value: 1,
                    prev: 3,
                    next: 1,
                },
                Node {
                    key: 2,
                    value: 3,
                    prev: 2,
                    next: 2,
                },
                Node {
                    key: 4,
                    value: 0,
                    prev: 0,
                    next: 1,
                },
            ],
        );
    }
}
