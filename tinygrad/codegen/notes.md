# Linearizer breakdown


linearizing is basically 'flatterning' the AST.
the Linearizer, which turns an AST into a list of (linear) UOps


UOps are the little ops that the CPU/GPU actually executes, e.g. LOOP, STORE, LOAD, BARRIER

```
UOps.LOOP           :                           []                               ([<ridx1[0-15]>], 'reduce')
UOps.LOAD           : <val1_0>                  []                               MemOp(i=1, idx=<((lidx0[0-255]*16)+ridx1[0-15])>, valid=<1>)
UOps.ALU            : <acc0_0>                  [<val1_0>, <acc0_0>]             BinaryOps.ADD
UOps.ENDLOOP        :                           []                               ([<ridx1[0-15]>], 'reduce')
```

Main output of the linearizer is a list of UOps, which is basically the 'lines of code'.

`LocalBuffer`:
- Basically a 'UOp' style thing that represents a thread-local array, e.g. 'temp' buffer in generated code.

`to_float4()`:
- takes a list of tokens and groups them into a single float4 token if they're:
  - all same
  - all have same name, and are consecutive with dtype==_float4)


`to_image_idx`:
 -> Takes a shape and an 'idxy' to convert it into a 'idx' and 'idy'


`Token`:
- individual unit that gets passed around representing something??


`get_grouped_float4_idxs()`:
- takes list of tokens, returns list of integer 'idxs'
- basically takes a list of tokens and returns groups of 4 that can be used as a float4 idx

`get_grouped_maybe_float4()`:
- takes list of tokens????????????

`expand_node()`:
- takes a single Node and turns it into a list of Nodes
- basically takes something like 'SumNode' and turns it into [Part, Part, Part]

`MemOp`:
- tiny little class with an i, idx, and valid

`UOp`:
- has: what UOp type (e.g. LOAD, STORE, LOOP, ENDLOOP, ALU, BARRIER)
- output token
- vinput token as list
- arg

`Linearizer`:
`__init__`:
- takes an AST (LazyOp) and a buffer we're going to output to.
- self.bufs just lists all buffers that are in play (all the ones in the ast, and our output)
  - self.bufs[0] is output
- self.key -> big string representing this AST for caching purposes


`process()`:
    `self.earlybufs` -> any buffer that occurs *before* the reduceop, e.g. it's inputs
    `self.sts` -> local clones of each of the shapetrackers for all of our buffers
    `self.sts[0]` -> output buffer shape, we auto resize immediately to be correct
    `self.full_buf_index` -> index of our first earlybuf?

    `self.full_shape` -> seems to be inputshape of some sort?
    `reduce` -> each individiual reduction that will occur
                -> e.g. each the transform for each individual axis that is going to happen
     `permute` ->
            -> first half is list of all the axis where no reduction will occur
            -> second half is list of all axis where a reduction will occur
            -> so essentially we move axis where a reduction will occur to the end.

      Does some setup
      calls
       self.simplify_ones()
       self.simplify_merge_adjacent()`


`shape_offsets()`: NO IDEA
`float4_axis()`: NO IDE

`upcasted_axis`:
  - takes a buffer index
  - returns axis sizes taht were upcasted
            axis strides taht were upcasted
            whether that axis matches the full shape

`acc_offsets`:
  - something to do with upcasted axis and their strides?????????????


`get_upcast_dim`:
  - takes a buffer index
  - checks if upcasting should occur (support float4, buffer is float dtype or image)
  - returns list of axis that should be upcasted?

`global_load`:
- creates the tokens for a 'load' operation???????

`global_store`:
- creates a store mem op sort of deal?

`LINEARIZE()`:



















