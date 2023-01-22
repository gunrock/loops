[README](/README.md) > **Abstraction**

# Abstraction

The simple idea behind our load-balancing abstraction is to represent sparse formats as atoms, tiles and set functional abstraction elements described in the "Function and Set Notation" below. Once represented as such, we can develop load-balancing algorithms that create balanced ranges of atoms and tiles and map them to processor ids. This information can be abstracted to the user with a simple API (such as ranged-for-loops) to capture user-defined computations. Some benefits of this approach are: (1) the user-defined computation remains largely the same for many different static or dynamic load-balancing schedules, (2) these schedules can now be extended to other computations and (3) dramatically reduces code complexity.

## As function and set notation.

Given a sparse-irregular problem $S$ made of many subsets called tiles, $T$. $T_i$ is defined as a collection of atoms, where an atom is the smallest possible processing element (for example, a nonzero element within a sparse-matrix). Using a scheduler, our abstraction's goal is to create a new set, $M$, which maps the processor ids (thread ids for a given kernel execution) $P_{id}$ to a group of subsets of $T$: $M = \{ P_{id}, T_i ... T_j \}$, map of processor ids to tiles, and the scheduler responsible for creating the maps: $L(S) = \{ M_0, ..., M_m\}$.

## As three domains: data, schedule and computation.

![illustration](https://user-images.githubusercontent.com/9790745/168728299-6b125b44-894a-49bb-92fd-ee85aaa80ae4.png)