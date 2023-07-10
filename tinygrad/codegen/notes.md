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
- main function that does the linearization
- 1. add local buffer for multistage reduce
    - adds the 'temp' Local buffer
    - only if 'len(self.group_for_reduce)' >= 1
  
- 2. defines local buffers for all 'local_alias' buffers

- 3. adds a global loop 'for i in range(0, self.first_reduce-self.local_dims)
     ? What are `self.first_reduce and self.local_dims`?
      - first reduce is property:
         -> first axis where a reduction will occur
      - local_dims is prop that is 
           self.first_reduce - global_count ??

- 4. adds a local loop
                   for i in range(self.first_reduce-self.local_dims, self.first_reduce+len(self.group_for_reduce)) -> so continues from above
- 5. upcast indexes
       - gets everything after self.upcasted for both `self.full_shape` and `self.output_shape`
        - ??? What are self.full_shape and self.output_shape really???
         - self.full_shape = self.sts[self.full_buf_index].shape
         - self.output_shape = self.sts[0].shape
         
         - self.shape_len = len(self.sts[0].shape) -> in theory all shapes are equal length then?
         - self.upcasted 
              -> we add one to it when we call .upcast
              -> seems to 'drop the last axis/dimension'?

-  6. make the reduce op
     1. define an accumulator
     2. define loop
     3. compute local aliases -> something to do with tensor cores
     4. load earlybufs
     5. run ast_parse
        -> takes a chunk of the ast and turns it into a list of tokens
     6. ends reduce loop
   
    7. if group_for_reduce -> doing local reduce
     - store in buffer
     - add a barrier
     - end loop

    8. 

`shift_to()`:
 - seems to be basically the same as what I was trying to do...
   moves a bunch of an axis to be somewhere elese 


`simplify_ones()`:
 - basically just removes axis where the size is 1 for all shapes being tracked
 - e.g. [1, 2, 3, 1, 1] -> [2, 3]

 `simplify_merge_adjacent()`:
 - not quite sure, something fancy to my knowledge


`required_optimizations()`:
- iterates through all buffers
  - lists all the axis with stride=1 and axis size divisible by 4
  - if we're an image or something?
     - basically attemtps to group axis together as 4 if it can?

`limit_global_dims`:
- attempts to merge dimensions togther if too many overall


`alias_buffer`
-> some fancy thing???


`hand_coded_optimizations`:
 - couple random parts to images
 - check if float4 not in use and first_reduce is early-ish (i.e. not too many other dims) and overall elements in first set of dimensions isn't too small
    -> does some weird shit where it moves stuff around for 'grouping'???
  - if not grouping, more optimizations
    - while elements impacted before first reduce is large
    -



















-> Upcasting seems to just be operations like converting 4 floats into a float4
-> Loop unrolling seems to also be an upcast?
-> 






































