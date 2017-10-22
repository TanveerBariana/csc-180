(deffacts Sacramento
	(start sac)
 	(dest davis)
)
(deffacts Chico
	(start chico)
)
(deffacts Galt
	(start galt)
 	(dest sac)
)
(deffacts Tahoe
	(start tahoe)
 	(dest fair)
)
(deffacts Fairfeild
	(start fair)
 	(dest galt)
)

(defrule p1
	?p <- (start)
 	=>
	(printout t "where do you start")
 	(assert (start (read)))
	(printout t "where do you end")
	(assert (end (read)))
 	(retract ?p)
)

(defrule rSac
	(Sacramento)
	=>
	(printout t "drive from sacramento to davis")
	(retract dest)
)
(defrule rChico
	(Chico)
	(dest Davis)
	=>
	(printout t "drive from chico to davis")
	(retract dest)
)
(defrule rChico
	(Chico)
	(dest galt)
	=>
	(printout t "drive from chico to Galt")
(defrule rFair
	(Fairfeild)
	=>
	(printout t "drive from Fairfeild to Galt")
	(retract dest)
	(retract start)
	(assert (Galt))
)
(defrule rGalt
	(Galt)
	=>
	(printout t "drive from Galt to Sacramento")
	(retract dest)
	(retract start)
	(assert (Sacramento))
)
(defrule rTahoe
	(Tahoe)
	=>
	(printout t "drive from tahoe to fairfeild")
	(retract dest)
	(retract start)
	(assert (Fairfeild))
)
(deffacts startup
 	(start)
)