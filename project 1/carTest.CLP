(defrule p1 
?p <- (start)
 =>
(printout t "engine turns? ")
 (assert (turns (read)))
 (retract ?p))
(defrule p2
 (turns no)
 =>
 (printout t "lights work? ")
 (assert (lights (read))))
(defrule rule2
 (turns no)
 (lights no)
 =>
 (printout t "problem is battery" crlf))
(defrule rule3
 (turns no)
 (lights yes)
 =>
 (printout t "problem is starter" crlf))
(defrule p5
 (turns yes)
 =>
 (printout t "tank has gas? ")
 (assert (tgas (read))))
(defrule p6
 (turns yes)
 (tgas yes)
 =>
 (printout t "carb has gas? ")
 (assert (cgas (read))))
(defrule rule4
 (tgas yes)
 (cgas yes)
=>
 (assert (egas yes)))
(defrule rule1
 (turns yes)
 (egas yes)
=>
 (printout t "problem is sparkplugs" crlf))
(deffacts startup
 (start))