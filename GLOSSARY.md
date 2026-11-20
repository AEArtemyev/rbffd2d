# RBF-FD

Vocabulary for neighbor selection and RBF-FD stencils in a planar domain.

## Language

**Planar domain**:
The two-dimensional region in which nodes are located and paths between them
are allowed to run.

**Polygonal domain**:
A planar domain described explicitly by polygonal boundary contours, including
contours of holes when present.

**Hole**:
A region enclosed by an inner boundary contour and excluded from the planar
domain.

**Direct visibility**:
Two points are directly visible to each other if their entire connecting
segment lies in the polygonal domain or on its boundary. Visibility from the
stencil center determines whether a node is eligible for its stencil.
_Avoid_: Unqualified neighbor when visibility eligibility is intended.

**Distance shell**:
The set of eligible nodes at the same Euclidean distance from a stencil center.
Neighbor selection and stencil expansion include or exclude each shell as a
whole.
_Avoid_: Boundary shell when it could be confused with the domain boundary.

**Intrinsic distance**:
The infimum of the lengths of paths connecting two points while staying within
the planar domain.
_Avoid_: Unqualified distance when it could mean either intrinsic or Euclidean distance.
