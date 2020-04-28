#![cfg_attr(not(test), no_std)]
#![feature(
    const_generics,
    const_int_pow,
    maybe_uninit_uninit_array,
    maybe_uninit_slice_assume_init,
    maybe_uninit_ref
)]

use core::borrow::Borrow;
use core::fmt;
use core::hash::{BuildHasher, Hash, Hasher};
use core::iter::ExactSizeIterator;
use core::mem::MaybeUninit;
use core::ptr;

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

    pub fn keys(&self) -> () {
        todo!()
    }

    pub fn values(&self) -> () {
        todo!()
    }

    pub fn values_mut(&mut self) -> () {
        todo!()
    }

    pub fn iter(&self) -> () {
        todo!()
    }

    pub fn iter_mut(&mut self) -> () {
        todo!()
    }
}

impl<K, V, S, const LEN: usize> HashMap<K, V, S, LEN>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    pub fn entry(&mut self, key: K) -> () {
        todo!()
    }

    pub fn get<Q: ?Sized>(&self, k: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        todo!()
    }

    pub fn get_mut<Q: ?Sized>(&mut self, k: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        todo!()
    }

    pub fn get_key_value<Q: ?Sized>(&self, k: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        todo!()
    }

    pub fn get_key_value_mut<Q: ?Sized>(&mut self, k: &Q) -> Option<(&K, &mut V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        todo!()
    }

    pub fn contains_key<Q: ?Sized>(&self, k: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        todo!()
    }

    pub fn update(&mut self, k: K, v: V) -> V {
        todo!()
    }

    pub fn try_update(&mut self, k: K, v: V) -> Result<V, V> {
        // returns v if k is not in this map else previous val
        todo!()
    }
}

#[cfg(test)]
mod test {
    use super::Node;
    type HashBuilder = core::hash::BuildHasherDefault<nohash_hasher::NoHashHasher<u32>>;
    type HashMap<const LEN: usize> = super::HashMap<u32, u32, HashBuilder, LEN>;

    #[test]
    fn test_equal_elemnts() {
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
