# fixed-hashmap

An always fully populated fixed size hashmap. You can think of it like an array but as a hashmap.
It's relation to `std::collections::HashMap<K, V>` is similar to the relation of `[T; LEN]` to `Vec<T>`.

This HashMap is fully allocated on the stack and therefore doesn't make any use of heap allocations.
It's also very easy to overflow the stack with big maps as each entry currently takes up at least 
`size_of::<Key> + size_of::<Key> + 2 * size_of::<usize>` bytes.

This crate relies on const generics so a nightly compiler is currently required.

## Example


```rust
use fixed_hashmap::HashMap;

let mut map = HashMap::<_, _, ahash::AHasher, 4>::new([("foo", 51), ("bar", 15), ("baz", 7), ("qux", 82)].iter().copied()).expect("iterator size was not 4");
map["bar"] = 63;
assert_eq!(map.get("foo"), Some(61));
assert_eq!(map.get("foobar"), None);
```

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

The Rust logo is owned by Mozilla and distributed under the terms of the
[Creative Commons Attribution license (CC-BY)](https://creativecommons.org/licenses/by/4.0/).

### Contribution


Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.