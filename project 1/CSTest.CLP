(defrule p1	?p <- (start)
	=>
	(printout t "CT or T " crlf)
	(assert (team (read)))
	(retract ?p))
(defrule CTLv1
	(team CT)
 =>
	(printout t "is it Round 1" crlf)
	(assert (r1 (read))))
(defrule CTLv2
	(team CT)
	(r1 yes)
	=>	(print t "buy D" crlf)	(print t "buy smoke")	(retract r1))
(defrule CTLv3
	(team CT)
	(r1 no)
	=>
	(printout t "is it Round 15" crlf)	(assert(r15 (read)))	(retract r1))
(defrule CTLv4
	(team CT)	(r15 yes)
	=>	(assert (fullbuyBreak both1))	(printout t "do you have armor" crlf)	(assert (Arm (read)))	(retract r15))
(defrule CTLv4
	(team CT)
	(r15 no)
	=>
	(printout t "Did you win last Round? "crlf)
	(assert (win (read)))	(retract r15))
(defrule CTLv5
	(team CT)
	(win yes)
	=>	(assert (fullbuyBreak both1))	(printout t "do you have armor" crlf)	(assert (Arm (read))))(defrule CTLv5	(team CT)	(win no)	=>	(printout t "Is it Round 2?" crlf)	(assert (r2 (read)))	; do yes(assert (egas yes)))(defrule CTLv6	(team CT)	(r2 yes)	=>	(assert (fullbuyBreak both1))	(printout t "do you have armor" crlf)	(assert (Arm (read)))  )(defrule CTLv6	(team CT)	(r2 no)	=>	(printout t "Is it *2" crlf)	(assert (star2 (read))))(defrule CTLv7	(team CT)	(star2 yes)	=>	;do fullbuy)(defrule CTLv7	(team CT)	(star2 no)	=>	(printout t "Can the enemy buy twice?" crlf)	(assert (ceb2 (read))))(defrule CTLv8	(team CT)	(ceb2 yes)	=>	(printout t "if you buy a pistol will you have more than $999.99 " crlf)	(assert (money (read))))(defrule CTLv8	(team CT)	(ceb2 no)	=>	;do halfbuy)(defrule CTLv9	(team CT)	(money yes)	=>	(printout t "buy a pistole"))(defrule CTLv9	(team CT)	(money no)	=>	(printout t "Save your money"));-------Full Buy ---------------------------------------------------(defrule FullBuy1	(fullbuyBreak both1)	(Arm no)	=>	(print t "buy armor" crlf)	(print t "do you have D")	(assert (D (read)))	(retract Arm))(defrule FullBuy1	(fullbuyBreak both1)	(D no)	(team CT)	=>	(print t "buy D" crlf)	(print t "do you have a rifle")	(assert (rif (read)))	(retract D))(defrule FullBuy1	(fullbuyBreak both1)	(rif no)	=>	(print t "can you afford a rifle" crlf)	(assert (rbuy (read)))	(retract rif))(defrule FullBuy1	(fullbuyBreak both1)	(rbuy yes)	=>	(print t "buy rifle" crlf)	(retract fullbuyBreak)	(assert (fullbuyBreak both2))	(retract rbuy))(defrule FullBuy1	(fullbuyBreak both1)	(rbuy no)	=>	(print t "do you have a pistol" crlf)	(assert (pbuy (read)))	(retract rbuy))(defrule FullBuy1	(fullbuyBreak both1)	(pbuy no)	=>	(print t "buy a pistol" crlf)	(retract fullbuyBreak)	(assert (fullbuyBreak both2))	(retract pbuy))(defrule FullBuy2	(fullbuyBreak both2)	=>	(printout t "buy smoke, flash, fire, flash in that order as far as you can")	(retract fullbuyBreak)	(assert (pass yes)));--------------------------------------------------------
(deffacts startup
 (start))