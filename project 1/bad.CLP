;(assert (from sacramento to davis))
;(assert (from chico to davis))
;(from fairfield to galt)
;(from galt to sacramento)
;(from tahoe to fairfield)
;(from chico to galt)

(deffacts initial-state
 (block A) (block B) (block C) (block D) (block E) (block F)
 (on-top-of nothing A)
 (on-top-of A B)
 (on-top-of B C)
 (on-top-of C floor)
 (on-top-of nothing D)
 (on-top-of D E)
 (on-top-of E F)
 (on-top-of F floor)
 (move-goal C on-top-of E))
(defrule move-directly
 ?goal <- (move-goal ?block1 on-top-of ?block2)
 (block ?block1)
 (block ?block2)
 (on-top-of nothing ?block1)
 ?stack-1 <- (on-top-of ?block1 ?block3)
 ?stack-2 <- (on-top-of nothing ?block2)
 =>
 (retract ?goal ?stack-1 ?stack-2)
 (assert (on-top-of ?block1 ?block2))
 (assert (on-top-of nothing ?block3))
 (printout t ?block1 "moved on top of" ?block2 "." crlf))
(defrule move-to-floor
 ?goal <- (move-goal ?block1 on-top-of floor)
 (block ?block1)
 (on-top-of nothing ?block1)
 ?stack <- (on-top-of ?block1 ?block2)
 =>
 (retract ?goal ?stack)
 (assert (on-top-of ?block1 floor))
 (assert (on-top-of nothing ?block2))
 (printout t ?block1 "moved on top of floor." crlf))
(defrule clear-upper-block
 (move-goal ?block1 on-top-of ?)
 (block ?block1)
 (on-top-of ?block2 ?block1)
 (block ?block2)
 =>
 (assert (move-goal ?block2 on-top-of floor)))
(defrule clear-lower-block
 (move-goal ? on-top-of ?block1)
 (block ?block1)
 (on-top-of ?block2 ?block1)
 (block ?block2)
 =>
 (assert (move-goal ?block2 on-top-of floor)))