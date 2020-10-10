#![cfg_attr(not(test), no_std)]
#![feature(min_const_generics, maybe_uninit_uninit_array, maybe_uninit_ref)]

use core::borrow::Borrow;
use core::fmt;
use core::hash::{BuildHasher, Hash, Hasher};
use core::iter::{ExactSizeIterator, FusedIterator};
use core::mem::{ManuallyDrop, MaybeUninit};
use core::ops;
use core::ptr;
use core::slice;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IterSizeMismatch;

impl fmt::Display for IterSizeMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the iterator length did not match up with the HashMap size"
        )
    }
}

fn hash_key<K: Hash + ?Sized>(hash_builder: &impl BuildHasher, key: &K) -> u64 {
    let mut state = hash_builder.build_hasher();
    key.hash(&mut state);
    state.finish()
}

#[derive(Clone)]
pub struct HashMap<K, V, S, const LEN: usize> {
    hash_builder: S,
    table: Table<K, V, LEN>,
}

impl<K, V, S, const LEN: usize> HashMap<K, V, S, LEN>
where
    K: Hash,
    S: BuildHasher + Default,
{
    #[inline]
    pub fn new<I>(iter: I) -> Result<Self, IterSizeMismatch>
    where
        I: IntoIterator<Item = (K, V)>,
        I::IntoIter: ExactSizeIterator,
    {
        Self::with_hasher(iter, S::default())
    }
}

impl<K, V, S, const LEN: usize> HashMap<K, V, S, LEN>
where
    K: Hash,
    S: BuildHasher,
{
    pub fn with_hasher<I>(iter: I, hash_builder: S) -> Result<Self, IterSizeMismatch>
    where
        I: IntoIterator<Item = (K, V)>,
        I::IntoIter: ExactSizeIterator,
    {
        let table = Table::new(&hash_builder, iter)?;
        Ok(HashMap {
            hash_builder,
            table,
        })
    }

    pub fn hasher(&self) -> &S {
        &self.hash_builder
    }
}

impl<K, V, S, const LEN: usize> HashMap<K, V, S, LEN> {
    #[inline]
    pub fn keys(&self) -> Keys<'_, K, V> {
        Keys { inner: self.iter() }
    }

    #[inline]
    pub fn values(&self) -> Values<'_, K, V> {
        Values { inner: self.iter() }
    }

    #[inline]
    pub fn values_mut(&mut self) -> ValuesMut<'_, K, V> {
        ValuesMut {
            inner: self.iter_mut(),
        }
    }

    #[inline]
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            inner: self.table.0.iter(),
        }
    }

    #[inline]
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
    #[inline]
    pub fn get<Q: ?Sized>(&self, k: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.get_key_value(k).map(|(_, v)| v)
    }

    #[inline]
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
            .find(hash, k)
            .map(|node| (&node.key, &node.value))
    }

    pub fn get_key_value_mut<Q: ?Sized>(&mut self, k: &Q) -> Option<(&K, &mut V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let hash = hash_key(&self.hash_builder, k);
        self.table
            .find_mut(hash, k)
            .map(|node| (&node.key, &mut node.value))
    }

    #[inline]
    pub fn contains_key<Q: ?Sized>(&self, k: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.get(k).is_some()
    }
}

impl<K: fmt::Debug, V: fmt::Debug, S, const LEN: usize> fmt::Debug for HashMap<K, V, S, LEN> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K, V, S, const LEN: usize> PartialEq for HashMap<K, V, S, LEN>
where
    K: Eq + Hash,
    V: PartialEq,
    S: BuildHasher,
{
    fn eq(&self, other: &Self) -> bool {
        self.iter()
            .all(|(k, v)| other.get(k).map_or(false, |v2| v == v2))
    }
}

impl<K, V, S, const LEN: usize> Eq for HashMap<K, V, S, LEN>
where
    K: Eq + Hash,
    V: Eq,
    S: BuildHasher,
{
}

impl<K, Q: ?Sized, V, S, const LEN: usize> ops::Index<&'_ Q> for HashMap<K, V, S, LEN>
where
    K: Eq + Hash + Borrow<Q>,
    Q: Eq + Hash,
    S: BuildHasher,
{
    type Output = V;
    #[inline]
    fn index(&self, key: &Q) -> &Self::Output {
        self.get(key).expect("no entry found for key")
    }
}

// hashmaps in rust currently dont implement this due to the possibility of IndexSet being a thing in the future
// this map does not support inserting after creation so its fine for us to implement
impl<K, Q: ?Sized, V, S, const LEN: usize> ops::IndexMut<&'_ Q> for HashMap<K, V, S, LEN>
where
    K: Eq + Hash + Borrow<Q>,
    Q: Eq + Hash,
    S: BuildHasher,
{
    #[inline]
    fn index_mut(&mut self, key: &Q) -> &mut Self::Output {
        self.get_mut(key).expect("no entry found for key")
    }
}

impl<'a, K, V, S, const LEN: usize> IntoIterator for &'a HashMap<K, V, S, LEN> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    #[inline]
    fn into_iter(self) -> Iter<'a, K, V> {
        self.iter()
    }
}

impl<'a, K, V, S, const LEN: usize> IntoIterator for &'a mut HashMap<K, V, S, LEN> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;

    #[inline]
    fn into_iter(self) -> IterMut<'a, K, V> {
        self.iter_mut()
    }
}

impl<K, V, S, const LEN: usize> IntoIterator for HashMap<K, V, S, LEN> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V, LEN>;

    #[inline]
    fn into_iter(self) -> IntoIter<K, V, LEN> {
        IntoIter {
            table: ManuallyDrop::new(self.table),
            index: 0,
        }
    }
}

// fixed size arrays in rust dont currently implement IntoIterator so we have to implement that ourselves
pub struct IntoIter<K, V, const LEN: usize> {
    table: ManuallyDrop<Table<K, V, LEN>>,
    index: usize,
}

impl<K, V, const LEN: usize> Iterator for IntoIter<K, V, LEN> {
    type Item = (K, V);

    fn next(&mut self) -> Option<(K, V)> {
        if self.index >= LEN {
            None
        } else {
            // SAFETY: We track what has been copied out through the index making sure to not cause a double drop
            //  when IntoIter drops
            let node: Node<_, _> = unsafe { ptr::read(&self.table.0[self.index] as *const _) };
            self.index += 1;
            Some((node.key, node.value))
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (LEN - self.index, Some(LEN - self.index))
    }
}

impl<K, V, const LEN: usize> Drop for IntoIter<K, V, LEN> {
    fn drop(&mut self) {
        for ele in &mut self.table.0[self.index..] {
            // SAFETY: we know that all the elements starting from `index` have not been returned through the
            //  iterator so we can safely drop them here.
            unsafe { ptr::drop_in_place(ele) };
        }
    }
}

impl<K, V, const LEN: usize> FusedIterator for IntoIter<K, V, LEN> {}
impl<K, V, const LEN: usize> ExactSizeIterator for IntoIter<K, V, LEN> {
    #[inline]
    fn len(&self) -> usize {
        LEN - self.index
    }
}

impl<K: fmt::Debug, V: fmt::Debug, const LEN: usize> fmt::Debug for IntoIter<K, V, LEN> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(Iter {
                inner: self.table.0.iter(),
            })
            .finish()
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

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                self.inner.next().map($closure)
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                self.inner.size_hint()
            }
        }

        impl<'a, K, V> FusedIterator for $ty {}
        impl<'a, K, V> ExactSizeIterator for $ty {
            #[inline]
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

#[derive(Clone)]
struct Table<K, V, const LEN: usize>([Node<K, V>; LEN]);

struct TableInitGuard<K, V, const LEN: usize> {
    table: [MaybeUninit<Node<K, V>>; LEN],
    init_map: [bool; LEN],
}

impl<K, V, const LEN: usize> TableInitGuard<K, V, LEN> {}

impl<K, V, const LEN: usize> Drop for TableInitGuard<K, V, LEN> {
    fn drop(&mut self) {
        for (entry, _) in self
            .table
            .iter_mut()
            .zip(self.init_map.iter())
            .filter(|(_, &initialized)| initialized)
        {
            // SAFETY: We track what elements are initialized so this is safe to drop
            unsafe { ptr::drop_in_place(entry.as_mut_ptr()) };
        }
    }
}

impl<K, V, const LEN: usize> Table<K, V, LEN>
where
    K: Hash,
{
    fn new<I>(hash_builder: &impl BuildHasher, iter: I) -> Result<Self, IterSizeMismatch>
    where
        I: IntoIterator<Item = (K, V)>,
        I::IntoIter: ExactSizeIterator,
    {
        let iter = iter.into_iter();
        if iter.len() != LEN {
            return Err(IterSizeMismatch);
        }

        let mut guard: TableInitGuard<K, V, LEN> = TableInitGuard {
            table: MaybeUninit::uninit_array(),
            init_map: [false; LEN],
        };
        let table = {
            let (table, init_map) = (&mut guard.table, &mut guard.init_map);

            for (key, value) in iter {
                let slot = hash_key(hash_builder, &key) as usize % LEN;
                // already occupied
                if init_map[slot] {
                    let next_free_slot = (0..LEN)
                        .map(|i| i + slot)
                        .find(|&idx| !init_map[idx % LEN])
                        .expect("bug: array is full");
                    // SAFETY: we checked that the slot is initialized
                    let (prev, next) = unsafe {
                        let slot = table[slot].assume_init_ref();
                        (slot.prev, slot.next)
                    };
                    // make room for new node, put previous in a free spot
                    table.swap(slot, next_free_slot);
                    // then write new node into the uninitialized memory and fix links
                    // A head, prepend this new node to the list
                    if prev == slot {
                        table[slot] = MaybeUninit::new(Node {
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
                                table[next_free_slot].assume_init_mut().next = next_free_slot;
                            // otherwise fix links due to element move
                            } else {
                                table[next].assume_init_mut().prev = next_free_slot;
                                table[next_free_slot].assume_init_mut().prev = slot;
                            }
                        }
                    // Not a head, this new node becomes a new list
                    } else {
                        table[slot] = MaybeUninit::new(Node {
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
                            table[prev].assume_init_mut().next = next_free_slot;
                            // slot was a tail
                            if next == slot {
                                // uphold tail's invariant of pointing to itself
                                table[next_free_slot].assume_init_mut().next = next_free_slot;
                            // slot was in the middle of a list
                            } else {
                                // update invalidated back pointer of the previous second element
                                table[next].assume_init_mut().prev = next_free_slot;
                            }
                        }
                    }

                    init_map[next_free_slot] = true;
                // free spot yay
                } else {
                    table[slot] = MaybeUninit::new(Node {
                        key,
                        value,
                        // is head
                        prev: slot,
                        // is tail
                        next: slot,
                    });
                    init_map[slot] = true;
                }
            }
            debug_assert!(init_map.iter().copied().all(core::convert::identity));
            // SAFETY: the array has been properly initialized
            //  Is this the current best way to turn a [MaybeUninit<T>; _] into a [T; _]?
            Table(unsafe { ptr::read(table as *mut _ as *mut [Node<K, V>; LEN]) })
        };
        // no panic happened, so don't run its Drop impl
        core::mem::forget(guard);
        Ok(table)
    }

    fn find<Q: ?Sized>(&self, hash: u64, key: &Q) -> Option<&Node<K, V>>
    where
        K: Borrow<Q>,
        Q: Eq,
    {
        let mut last = (hash % LEN as u64) as usize;
        let mut entry = &self.0[last];
        loop {
            if entry.key.borrow() == key {
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

    fn find_mut<Q: ?Sized>(&mut self, hash: u64, key: &Q) -> Option<&mut Node<K, V>>
    where
        K: Borrow<Q>,
        Q: Eq,
    {
        let mut entry_slot = (hash % LEN as u64) as usize;
        let mut entry = &self.0[entry_slot];
        loop {
            if entry.key.borrow() == key {
                break Some(entry_slot);
            }
            if entry.next != entry_slot {
                entry_slot = entry.next;
                entry = &self.0[entry.next];
            } else {
                break None;
            }
        }
        // borrowck doesnt allow us to early return a mut ref, as this would exclusively borrow it for the rest of the function
        //  so we have to break out the slot and then reborrow with that
        .map(move |slot| &mut self.0[slot])
    }
}

impl<K: fmt::Debug, V: fmt::Debug, const LEN: usize> fmt::Debug for Table<K, V, LEN> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Table").field(&&self.0[..]).finish()
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(test, derive(PartialEq))]
struct Node<K, V> {
    key: K,
    value: V,
    // if pref == self -> head
    // if pref == self && next == self -> single
    prev: usize,
    // if next = self -> tail
    next: usize,
}

#[cfg(test)]
mod test {
    use super::Node;
    type HashBuilder = core::hash::BuildHasherDefault<nohash_hasher::NoHashHasher<u32>>;
    type HashMap<const LEN: usize> = super::HashMap<u32, u32, HashBuilder, LEN>;

    #[test]
    fn test_get() {
        let foo = HashMap::<5>::new(vec![(0, 0), (3, 3), (5, 5), (8, 8), (1, 1)]).unwrap();

        assert_eq!(foo.get(&0), Some(&0));
        assert_eq!(foo.get(&1), Some(&1));
        assert_eq!(foo.get(&3), Some(&3));
        assert_eq!(foo.get(&5), Some(&5));
        assert_eq!(foo.get(&8), Some(&8));
    }

    #[test]
    fn test_equal_elements() {
        // FIXME should this error, or is this fine and considered a logical error?
        let foo = HashMap::<2>::new(vec![(0, 0), (0, 0)]).unwrap();

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
        let foo = HashMap::<2>::new(vec![(0, 0), (2, 2)]).unwrap();

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
        let foo = HashMap::<3>::new(vec![(0, 1), (3, 0), (4, 2)]).unwrap();

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
        let foo = HashMap::<3>::new(vec![(0, 1), (3, 0), (6, 2)]).unwrap();

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
        let foo = HashMap::<4>::new(vec![(0, 1), (4, 0), (8, 2), (2, 3)]).unwrap();

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

    #[test]
    fn test_init_guard() {
        use std::sync::Arc;
        let count = Arc::new(());

        let _ = std::panic::catch_unwind(|| {
            super::HashMap::<u32, Arc<()>, nohash_hasher::BuildNoHashHasher<u32>, 10>::new(
                vec![
                    (0, Arc::clone(&count)),
                    (1, Arc::clone(&count)),
                    (2, Arc::clone(&count)),
                    (3, Arc::clone(&count)),
                    (4, Arc::clone(&count)),
                    (5, Arc::clone(&count)),
                    (6, Arc::clone(&count)),
                    (7, Arc::clone(&count)),
                    (8, Arc::clone(&count)),
                    (9, Arc::clone(&count)),
                ]
                .into_iter()
                .inspect(|&(k, _)| {
                    if k == 6 {
                        panic!()
                    }
                }),
            )
            .unwrap();
        });
        assert_eq!(Arc::strong_count(&count), 1);
    }
}
