"""Apply hand-written gold labels to golden_sample_to_label.csv -> golden_eval_set.csv.

Labels are stored here as a dict keyed by row_id, each with a short verbatim
`check` substring from that row's customer_text. Before writing anything, we
assert the substring actually appears in the row it claims to label -- this
catches row-id misalignment immediately (loudly) instead of silently writing
a mislabeled gold set. See LABELING_GUIDE.md for the rubric used.
"""
from __future__ import annotations

import pandas as pd

from agent import config

# row_id -> (check_substring, gold_intent, gold_escalate, gold_escalation_reason, notes)
LABELS: dict[int, tuple[str, str, bool, str, str]] = {}


def add(row_id: int, check: str, intent: str, escalate: bool, reason: str, note: str) -> None:
    LABELS[row_id] = (check, intent, escalate, reason, note)


# ---- batch 1: rows 1-40 -----------------------------------------------
add(1, "3 forms", "complaint_escalation", True,
    "vague repeated-effort complaint, no resolvable topic stated",
    "ask what the actual unresolved issue is rather than requesting another form")
add(2, "Royal Mail", "order_delivery", False,
    "routine delivery-carrier preference question", "explain carrier options if available")
add(3, "now it's not coming until Monday", "order_delivery", False,
    "routine delivery-estimate question, standard transit-time explanation applies",
    "explain dispatch vs transit time")
add(4, "box of stuff that I didn't order", "order_delivery", False,
    "unexpected/wrong item, no charge dispute stated, routine clarification",
    "ask if it could be a gift; offer contact channel")
add(5, "ruining my 3 year olds Christmas", "order_delivery", False,
    "single damaged-item report, standard remedy flow", "offer replacement/refund, empathetic tone")
add(6, "condition", "returns_refund", True,
    "customer disputes brand's refusal to exchange a broken item",
    "needs human judgment on the policy exception")
add(7, "disappointed by Amazon return service", "returns_refund", False,
    "vague dissatisfaction, no specifics yet", "ask what specifically went wrong with the return")
add(8, "Alexa app in Indian play store", "product_inquiry", False,
    "routine app-availability question", "state availability/timeline")
add(9, "refurbished one", "returns_refund", True,
    "customer disputes quality of replacement (refurbished vs new)",
    "needs human judgment on swapping for a new unit")
add(10, "9-5pm", "order_delivery", False,
    "single-instance delay complaint", "ask for tracking status, acknowledge the wait")
add(11, "complaint emails", "complaint_escalation", True,
    "references a prior complaint that was mishandled",
    "needs human to review what happened to the original complaint")
add(12, "fire stick", "product_inquiry", False,
    "device malfunction question, standard flow", "ask for more detail, direct to device support")
add(13, "Amazon Echo users only or all prime members", "product_inquiry", False,
    "routine content-availability question", "answer eligibility if known")
add(14, "fucked up", "unclear_other", True,
    "no discernible topic, just hostility", "ask what's wrong without assuming a topic")
add(15, "log in issues right now", "account_access", False,
    "routine status check", "ask what error they're seeing")
add(16, "Phillips home hub", "product_inquiry", False,
    "routine compatibility question", "answer directly if known")
add(17, "courier person didn't even try", "order_delivery", False,
    "single courier complaint, standard flow", "apologize, offer to investigate the tracking scan")
add(18, "price match", "billing_payment", False,
    "simple price-match policy FAQ, no dispute", "state policy plainly")
add(19, "malayalam movies", "product_inquiry", False,
    "content-catalog availability question", "give honest status")
add(20, "0 made it", "order_delivery", False,
    "delivery-failure pattern in one sitting, no prior support-contact stated",
    "ask for carrier/order specifics")
add(21, "cancel a pre-order", "returns_refund", False,
    "routine cancellation self-service issue", "offer to help cancel")
add(22, "damaged at AMD1 FC", "order_delivery", False,
    "single request to flag damaged items to the team", "acknowledge and forward")
add(23, "Pethatic", "order_delivery", False,
    "Prime-value complaint framed around a late delivery, single instance",
    "address delivery timing")
add(24, "I have send you a message", "unclear_other", False,
    "references a prior DM with no topic given here", "confirm DM receipt/status")
add(25, "sucky service", "complaint_escalation", False,
    "vague general dissatisfaction, no specifics", "ask what specifically went wrong")
add(26, "get refund mail", "returns_refund", True,
    "conflicting information given (reshipped vs refunded)",
    "needs human to reconcile actual order status")
add(27, "never send bookmarks", "product_inquiry", False,
    "routine question about a promotional extra (bookmarks)", "explain bookmarks aren't guaranteed")
add(28, "prioritizing an order", "order_delivery", False,
    "vague order-prioritization complaint but has a clear order topic",
    "ask for order details despite hostile framing")
add(29, "leaving my parcel on the doorstep", "order_delivery", False,
    "single delivery-handling complaint, explicitly first occurrence",
    "apologize, check if contents were damaged")
add(30, "Thanksgiving dinner", "order_delivery", False,
    "single carrier-performance complaint", "acknowledge, no dispute stated")
add(31, "Miami cust", "order_delivery", False,
    "pattern of delivery failures, no prior support-contact history stated",
    "ask for current order details")
add(32, "broken since I upgraded to iOS 11", "product_inquiry", False,
    "app bug report after OS upgrade, not a delivery issue despite weak-label guess",
    "ask them to report the bug")
add(33, "still waiting on an order from over a week ago", "order_delivery", False,
    "single delay complaint", "ask for delivery-date confirmation")
add(34, "decreasingQuality", "order_delivery", False,
    "packaging-quality feedback tied to a delivery", "acknowledge feedback")
add(35, "£7.10", "billing_payment", True,
    "specific disputed charge amount for a phone call",
    "needs verification, do not confirm/deny without account access")
add(36, "smaller players in the market", "returns_refund", False,
    "general returns-process complaint, no specific case", "acknowledge feedback")
add(37, "resolutions team", "returns_refund", False,
    "customer already mid-resolution and hopeful, not requesting new action",
    "reassure, no new action needed")
add(38, "opposite", "order_delivery", False,
    "single bad-delivery-handling report", "apologize for handling")
add(39, "charged me twice WTF", "billing_payment", True,
    "concrete double-charge claim", "needs account access to verify and refund duplicate charge")
add(40, "stream reputation", "product_inquiry", False,
    "routine digital-content/library question", "explain AutoRip eligibility")

# ---- batch 2: rows 41-80 -----------------------------------------------
add(41, "Mountain Dew syrup", "billing_payment", True,
    "unauthorized purchase by a family member", "needs judgment on return options + payment-method security")
add(42, "thieving", "account_access", True,
    "account access issue plus hostility/accusation", "needs human due to hostility and lockout")
add(43, "keep being rescheduled", "order_delivery", False,
    "single delayed-shipment complaint despite expedited shipping", "acknowledge, check carrier status")
add(44, "nvr call back", "order_delivery", True,
    "customer states a prior support promise (48-hour resolution) was broken",
    "needs human because a prior commitment was already missed")
add(45, "power adpter stop working", "returns_refund", False,
    "single defective-device replacement/refund process question", "clarify replacement vs refund timing")
add(46, "Please send me a DM", "account_access", False,
    "routine account-help request, wants a DM channel", "offer DM/contact channel")
add(47, "on the 13th", "order_delivery", False,
    "single late-order complaint", "acknowledge and check status")
add(48, "slacking in the shipping department", "order_delivery", False,
    "single pattern complaint about lost packages", "ask for specifics on the most recent case")
add(49, "robbing me", "billing_payment", True,
    "large charge (£450) with an explicit theft accusation and undelivered item",
    "needs human due to amount + fraud-adjacent language")
add(50, "starts at random", "product_inquiry", False,
    "routine device-malfunction question", "clarify troubleshooting vs manufacturer warranty")
add(51, "peanut butter instead of a phone", "returns_refund", True,
    "refund was promised then denied — direct contradiction customer disputes",
    "needs human to resolve conflicting information")
add(52, "KYC", "order_delivery", False,
    "delivery/customs verification process complaint, single instance", "acknowledge process frustration")
add(53, "Call of Duty", "order_delivery", False,
    "single missing-package/tracking-mismatch report", "offer standard missing-package troubleshooting")
add(54, "left in the rain for 8 hours", "order_delivery", False,
    "single late/mishandled delivery report (sarcastic thanks)", "apologize, ask about condition of contents")
add(55, "Wedding Anniversary", "complaint_escalation", True,
    "multiple compounding grievances with a stated intent to stop being a customer",
    "needs human triage across several issues at once")
add(56, "best eCommerce site", "compliment", False,
    "customer is criticizing a different company and praising Amazon by comparison",
    "tricky intent-classification trap — flag in failure analysis")
add(57, "FREE HANDBAG", "compliment", False,
    "pure positive feedback", "thank the customer, no action needed")
add(58, "Diwali sale", "unclear_other", False,
    "marketing-timing feedback, not a personal support request", "acknowledge feedback")
add(59, "elderly father's bank", "account_access", True,
    "third-party (elderly family member) financial-account access — sensitive",
    "needs human verification before discussing account access")
add(60, "taken the money twice", "billing_payment", True,
    "concrete double-charge claim after a declined-payment email",
    "needs account access to verify and reverse duplicate charge")
add(61, "senior U.K. Legal people", "complaint_escalation", True,
    "explicit request for legal/senior-contact escalation", "route to escalation path, don't attempt to resolve directly")
add(62, "irritated with", "unclear_other", False,
    "no discernible topic beyond general irritation", "ask what's going on")
add(63, "408-5004540", "order_delivery", False,
    "has a specific order reference and delivery/tracking complaint despite the opening rant",
    "investigate the wrong tracking info cited")
add(64, "ChanukahGift", "order_delivery", False,
    "mild, hopeful status mention, barely a complaint", "light acknowledgment")
add(65, "2 out of my 3 Amazon Prime orders", "order_delivery", False,
    "single missing-item-from-multi-item-order report", "ask if the other items shipped together")
add(66, "blocked my account twice", "account_access", True,
    "account blocked twice plus cancelled orders, with hostility",
    "needs human due to account-level action + repeated blocking")
add(67, "guaranteed", "order_delivery", False,
    "single missed-delivery-date complaint despite paid expedited shipping", "standard delivery-date check")
add(68, "asks verification code", "account_access", True,
    "locked out after email change, AND the official contact channel is broken for them",
    "self-serve and normal contact both failed; needs human")
add(69, "worstcustomerservice", "complaint_escalation", False,
    "vague general complaint, no specifics", "ask for details, single instance")
add(70, "fire hazard", "product_inquiry", True,
    "potential product safety/fire-hazard report", "safety issues need human/specialist review regardless of category")
add(71, "delivery boy. Let him verify", "unclear_other", False,
    "unsolicited process-improvement suggestion, not a personal issue", "thank for feedback, forward to team")
add(72, "prime music in india", "product_inquiry", False,
    "content-availability question framed positively", "give honest availability status")
add(73, "charged twice. Order no", "billing_payment", True,
    "concrete double-charge claim with order number", "needs account access to verify")
add(74, "parcel has arrived but it hasn't", "order_delivery", False,
    "single discrepancy between notification and reality", "standard missing-package flow")
add(75, "Jeff Bezos", "returns_refund", False,
    "single refund-status question, sarcastic but not extreme", "give refund timeline/status check")
add(76, "Dishonest delivery setup", "order_delivery", True,
    "pricing/cancellation dispute combined with a 'dishonest' accusation",
    "needs human to reconcile pricing/cancellation conflict")
add(77, "wwe2k18", "order_delivery", False,
    "routine delivery-date-change question", "explain the date change if info available")
add(78, "Guaranteed* delivery dates", "order_delivery", False,
    "single missed-guarantee complaint (sarcastic)", "standard delivery-guarantee explanation")
add(79, "tug of war", "returns_refund", False,
    "confusing but low-stakes cancellation question about an assisted order",
    "clarify why the order was cancelled")
add(80, "60 days for a refund", "returns_refund", True,
    "specific, severe refund delay (60 days) stated",
    "needs human — self-reported delay far exceeds normal SLA")

# ---- batch 3: rows 81-120 -----------------------------------------------
add(81, "delivered but you didn", "order_delivery", False,
    "single delivered-but-not-received report", "standard missing-package troubleshooting")
add(82, "priority email", "unclear_other", False,
    "niche pre-order/ticketing timing question, ambiguous topic", "ask for more detail on what was ordered")
add(83, "receiving this after 3 days", "order_delivery", False,
    "single delivery-speed complaint", "standard delay explanation/apology")
add(84, "account that", "account_access", True,
    "account locked, no self-serve path evident", "needs human/account specialist")
add(85, "1yr prime subscptin", "account_access", False,
    "routine subscription-timing question", "clarify renewal timing tied to shipment")
add(86, "does not equal 6 days", "order_delivery", False,
    "single Prime shipping-speed complaint", "standard shipping-time explanation")
add(87, "Baseball episodes", "billing_payment", False,
    "minor subscription-value complaint tied to a delayed bug fix, no dispute amount stated",
    "acknowledge, no refund demanded")
add(88, "Contact imdtly", "order_delivery", True,
    "explicit urgent demand ('contact immediately') plus item-not-received-despite-delivered-status",
    "urgency + status conflict warrants human follow-up")
add(89, "8 months", "account_access", True,
    "severe, long-standing (8 months) access issue", "self-serve clearly hasn't worked; needs human")
add(90, "supports fraud", "complaint_escalation", True,
    "explicit fraud accusation against the company", "serious accusations need human review")
add(91, "vivo power bank", "order_delivery", False,
    "single, specific wrong-item-delivered report", "offer return/replacement path")
add(92, "grocery store lines", "compliment", False,
    "pure thanks, no open request", "acknowledge warmly")
add(93, "scheduled pickup", "returns_refund", False,
    "single confusing pickup-cancellation report", "investigate the cancellation")
add(94, "gonna take 3 days to deliver my DVD", "order_delivery", False,
    "single delivery-timing question", "standard shipping-time explanation")
add(95, "couldn", "order_delivery", False,
    "single bad-delivery-experience report", "apologize, offer alternate-carrier note as feedback")
add(96, "web browser but the app allows", "account_access", False,
    "self-serve browser login issue, standard troubleshooting", "suggest clearing cookies/cache")
add(97, "charged for an item I don", "billing_payment", True,
    "customer claims they did not place the order they are being charged for",
    "needs human/fraud review, not routine order-timing question")
add(98, "Amazon Transportation Services", "order_delivery", False,
    "single delivery-contact-info complaint", "standard investigate-and-apologize")
add(99, "before 12:30 today", "order_delivery", False,
    "routine delivery-window question", "explain courier delivery hours")
add(100, "returnng mony", "returns_refund", True,
    "concrete claim that a refund/return of funds was never processed",
    "needs account access to verify and process refund")
add(101, "when my today order will", "order_delivery", False,
    "routine delivery-time question", "give standard delivery-window info")
add(102, "Intelcom", "order_delivery", False,
    "shipping-terms confusion, single instance", "explain 2-day shipping transit-time definition")
add(103, "CatEnrichment", "compliment", False,
    "pure positive feedback", "acknowledge warmly")
add(104, "bug or promotion", "product_inquiry", False,
    "website display bug report", "acknowledge, forward as a bug report")
add(105, "ontrac", "order_delivery", False,
    "conditional/future statement, not yet an actual dispute", "no action needed until described failure happens")
add(106, "release date of a book has changed", "product_inquiry", False,
    "feedback about release-date-change notification usefulness", "acknowledge feedback")
add(107, "pleasehelpamazon", "order_delivery", False,
    "routine delivery-status request", "ask for delivery-date confirmation")
add(108, "performing really bad lately", "product_inquiry", False,
    "single service-quality complaint about a specific feature (Alexa music)",
    "ask for device/specifics to troubleshoot")
add(109, "two months", "returns_refund", True,
    "two-month-old order, cancelled, refund never received", "needs human to locate the missing refund")
add(110, "Sodexo", "billing_payment", False,
    "simple payment-method policy question, no dispute", "state policy plainly")
add(111, "FireTV in India", "product_inquiry", False,
    "routine device/regional-availability question", "give honest availability status")
add(112, "CopyPaste", "order_delivery", True,
    "explicit meta-complaint about receiving low-quality templated responses to a delivery issue",
    "an automated-feeling reply would confirm the complaint; needs a human, personalized response")
add(113, "wishlist", "product_inquiry", False,
    "wishlist/gift feature confusion", "clarify how the gift/wishlist feature works")
add(114, "payment wasnt accepted", "billing_payment", True,
    "payment-processing failure blocking an order, not proactively notified",
    "needs account access to investigate payment failure")
add(115, "Who do I complain to", "order_delivery", True,
    "explicit request for an escalation/complaints channel", "provide the actual complaints/escalation path")
add(116, "return pickup is rescheduled", "returns_refund", False,
    "single return-pickup delay report", "standard pickup-delay flow")
add(117, "half and hour", "complaint_escalation", True,
    "documented ~30 minutes of unsuccessful phone contact plus a product defect",
    "normal channels already failed; needs human")
add(118, "Robert Plant", "order_delivery", False,
    "routine urgent delivery-timing request", "give delivery-window info; urgency alone isn't an escalation trigger")
add(119, "overcharged me for my order", "billing_payment", True,
    "overcharge dispute plus explicit request for a specific (human) representative",
    "explicit human-agent request")
add(120, "refusing to accept my honest review", "complaint_escalation", True,
    "accusation that the company is suppressing a negative review",
    "needs human/policy review of the review-moderation claim")

# ---- batch 4: rows 121-160 -----------------------------------------------
add(121, "Myntra", "order_delivery", False,
    "single delivery-timing comparison complaint", "standard delivery-window explanation")
add(122, "a few hours later they went on sale", "returns_refund", False,
    "commentary on a return/repurchase process, not requesting new action", "no action needed, informational")
add(123, "45 minutes on the phone", "billing_payment", True,
    "payment/gift-card dispute with documented 45-minute prior phone call",
    "prior extensive contact plus unresolved payment dispute")
add(124, "80 rupees extra", "billing_payment", True,
    "concrete extra-charge dispute tied to a broken delivery guarantee",
    "needs verification of guarantee terms and the extra charge")
add(125, "LAST FIRDAY", "order_delivery", False,
    "single missing-order report (sarcastic tone)", "standard status check")
add(126, "KaroMilkeLateDelivery", "order_delivery", False,
    "general late-delivery complaint", "standard delay acknowledgment")
add(127, "print from a Kindle", "product_inquiry", False,
    "routine device how-to question", "answer directly if documented")
add(128, "forgot to cancel my membership", "billing_payment", False,
    "self-acknowledged oversight (forgot to cancel), low conflict",
    "offer cancellation link, mention refund eligibility per policy")
add(129, "FOAM RUBBER", "unclear_other", False,
    "packaging-sustainability feedback, not a personal issue to resolve", "acknowledge feedback")
add(130, "fails to materialise", "order_delivery", False,
    "single missed-guarantee complaint", "standard apology/explanation")
add(131, "shuffle music from a playlist", "product_inquiry", False,
    "routine device how-to question", "answer directly or link help page")
add(132, "Order not fulfilled", "order_delivery", False,
    "single unfulfilled-order complaint", "ask for order status/explanation")
add(133, "Redmi 4A", "product_inquiry", False,
    "promo/sale visibility question", "give honest status of the sale")
add(134, "407-2602912", "order_delivery", False,
    "single lost-item replacement request tied to a prepaid order", "standard lost-item/replacement flow")
add(135, "it was damaged", "order_delivery", False,
    "single damaged-item-on-arrival report", "offer replacement/refund options")
add(136, "TreasureTruck", "compliment", False,
    "pure positive feedback", "acknowledge warmly")
add(137, "latest time a Prime delivery", "order_delivery", False,
    "routine delivery-window question", "give standard delivery-hours info")
add(138, "have to return an order", "returns_refund", False,
    "routine return-initiation difficulty", "offer to help start the return")
add(139, "Two parcels missing", "order_delivery", False,
    "same-day missing-package report (2 parcels), no prior-contact history stated",
    "standard missing-package investigation despite severity")
add(140, "proof of delivery or a full refund", "returns_refund", True,
    "explicit refund demand as an alternative to proof of delivery, after a week's wait",
    "concrete financial ask needing human follow-through")
add(141, "updates for echo dot", "product_inquiry", False,
    "routine device/content question", "answer directly")
add(142, "Seller Customer Service is the WORST", "complaint_escalation", False,
    "vague 'worst ever' complaint about a specific seller, no specifics given",
    "ask for details of the seller interaction")
add(143, "changed my password", "account_access", True,
    "potential unauthorized account activity prompting a security-motivated password change",
    "security concern needs human/account-specialist confirmation")
add(144, "2 times promised for return", "returns_refund", True,
    "two documented broken promises about a return pickup",
    "needs human — self-service promises already failed twice")
add(145, "home services team", "unclear_other", False,
    "vague general commentary about a services team, not a specific personal request",
    "ask for specifics")
add(146, "picture of it at my house", "order_delivery", False,
    "single delivered-but-missing report", "standard missing-package troubleshooting")
add(147, "lost my package twice in a row", "order_delivery", True,
    "explicit 'twice in a row' repeated-loss pattern stated",
    "documented repeat failure warrants human follow-up")
add(148, "3 times and have no positive resp", "returns_refund", True,
    "explicit '3 times' contact count stated with no resolution", "documented repeated-contact failure")
add(149, "colourblind", "unclear_other", False,
    "ambiguous product/image mixup, unclear specifics", "ask for a screenshot/more detail")
add(150, "brilliant packaging", "compliment", False,
    "pure positive feedback", "acknowledge warmly")
add(151, "4k + HDR", "product_inquiry", False,
    "routine content-availability question", "give honest availability status")
add(152, "video selection", "product_inquiry", False,
    "vague app-quality complaint", "ask for specifics of what's wrong")
add(153, "PS4 app complete garbage", "unclear_other", False,
    "vague app-quality remark with no specifics offered", "ask if facing specific issues with the app")
add(154, "surprise", "compliment", False,
    "pure positive feedback (delivered ahead of schedule)", "acknowledge warmly")
add(155, "gulcometer", "order_delivery", False,
    "single request for genuine/original product on a placed order", "acknowledge, offer contact path")
add(156, "kindke broken 2nd time", "product_inquiry", True,
    "explicit '2nd time' recurrence of the same device defect stated",
    "documented repeat product failure warrants closer look than a first-time apology")
add(157, "Alexa Voice Remote", "compliment", False,
    "pure positive feedback", "acknowledge warmly")
add(158, "D01-0572539", "returns_refund", True,
    "concrete money-at-risk claim (payment taken, item not received, refund also not given)",
    "needs human to resolve the dual money claim")
add(159, "Misleading ads", "order_delivery", False,
    "general complaint about ads/service quality, no specific order", "acknowledge feedback")
add(160, "taken payment for two items", "order_delivery", True,
    "repeated word 'again' implies a recurring failure, combined with a payment concern for two items",
    "pattern + payment concern warrants human review")

# ---- batch 5: rows 161-200 -----------------------------------------------
add(161, "chat team are useless", "product_inquiry", True,
    "device malfunction combined with an explicitly broken support channel (wrong number given) and stated anger",
    "normal support channel already failed for this customer")
add(162, "not even arrive when I need them", "order_delivery", False,
    "single Prime-value complaint", "standard delivery-window explanation")
add(163, "trying for last 2 hours", "product_inquiry", True,
    "explicit '2 hours' spent trying unsuccessfully to reach support",
    "documented repeated-attempt evidence")
add(164, "white noise machine", "product_inquiry", False,
    "routine device troubleshooting question", "ask for more detail on the connectivity issue")
add(165, "cheated n looted", "complaint_escalation", True,
    "serious 'cheated and looted' fraud-level accusation", "needs human review of the claim, not a generic apology")
add(166, "gift voucher", "billing_payment", True,
    "concrete gift-voucher activation dispute with a valid code confirmed but blocked by policy",
    "needs human to resolve the specific policy conflict")
add(167, "so lazy", "returns_refund", False,
    "single return-delay complaint framed as a rhetorical question", "standard pickup-delay flow")
add(168, "5-6 times", "order_delivery", True,
    "explicit '5-6 times' call count stated with no resolution",
    "documented extensive repeated-contact failure")
add(169, "Seller Account closed due to inactivity", "account_access", False,
    "routine account-status question (seller account closed)",
    "state policy plainly (cannot reopen, can open new account)")
add(170, "call the CEO of Amazon", "order_delivery", True,
    "explicit (if sarcastic) invocation of escalating to the CEO",
    "treat any executive/legal escalation language as a genuine escalation signal")
add(171, "Happening more", "order_delivery", False,
    "pattern complaint ('more and more') without a specific count or date",
    "standard delivery-window explanation despite frustration")
add(172, "wish lists disappeared", "product_inquiry", False,
    "feature/data-loss question (wishlist), not an account-security issue",
    "ask for account/sign-in confirmation, investigate feature bug")
add(173, "SuperMarioOdyssey", "order_delivery", False,
    "single unfulfilled-preorder report", "standard shipment-status check")
add(174, "quality of your watches is pathetic", "product_inquiry", False,
    "two separate defective-product reports but no money/contact-history/hostility trigger met",
    "acknowledge both defects, offer standard troubleshooting/return path")
add(175, "Rs 100", "billing_payment", True,
    "concrete refund amount dispute with shipment charge (Rs 100) specified",
    "needs account access to verify and process the specific refund")
add(176, "locked me out of my account", "account_access", True,
    "locked out with no self-serve path stated", "needs human/account specialist")
add(177, "next day delivery stinks", "order_delivery", False,
    "single delivery-experience complaint (sarcastic)", "standard delivery explanation")
add(178, "out for delivery since 9am", "order_delivery", False,
    "single urgent same-day delivery frustration, no dispute or repeat-contact stated",
    "acknowledge urgency, standard delivery-window info")
add(179, "in warranty now and need to repair", "order_delivery", False,
    "documentation/invoice request tied to an existing order; message contains a phone number and email",
    "provide invoice/support path; do not repeat the customer's PII in the reply")
add(180, "huge letdown", "order_delivery", False,
    "single delivered-but-not-received report", "standard missing-package troubleshooting")
add(181, "give them codes to our homes", "order_delivery", False,
    "single delivery-security-practice concern (door codes), informational",
    "acknowledge the concern about sharing door codes")
add(182, "resolution NOW", "complaint_escalation", True,
    "explicit 'NOW' demand for resolution combined with an unspecified underlying issue",
    "urgent + ambiguous is exactly the unsafe-to-auto-handle combination")
add(183, "1230 deducted", "billing_payment", True,
    "precise financial dispute with exact amounts calculated by the customer",
    "needs account access to verify the exact refund math")
add(184, "sync my last read page", "product_inquiry", False,
    "routine device troubleshooting question", "ask for troubleshooting steps already tried")
add(185, "7.99 repeating monthly charge", "billing_payment", False,
    "simple charge-identification question, not a dispute", "explain likely source of the recurring charge")
add(186, "50 refund", "returns_refund", True,
    "customer is disputing/pushing back on an already-offered settlement amount",
    "needs human because a negotiation is already underway")
add(187, "music keeps glitching", "product_inquiry", False,
    "routine device malfunction report", "ask for device/troubleshooting details")
add(188, "didn", "returns_refund", True,
    "specific date-stamped pending-refund claim plus a serious 'no money to refund' allegation",
    "needs human/finance review of the specific claim")
add(189, "Rotz", "unclear_other", True,
    "non-English message; agent is explicitly scoped to English only",
    "route to correct-language support rather than guess a reply")
add(190, "8668844954", "complaint_escalation", True,
    "explicit request for intervention plus a phone number provided and stated repeated contact",
    "explicit intervention request with contact history")
add(191, "44 phone calls", "returns_refund", True,
    "extreme, specific documented repeated-contact count", "one of the clearest escalate cases in the set")
add(192, "2 days shipping", "order_delivery", False,
    "routine shipping-terms confusion", "explain shipping transit-time definition")
add(193, "out for delivery", "order_delivery", False,
    "single undelivered-item-despite-notification report", "standard missing-package troubleshooting")
add(194, "primesucks", "order_delivery", False,
    "single unfulfilled-order complaint with policy framing", "standard status check despite frustrated tone")
add(195, "SayNoToAmazon", "order_delivery", True,
    "shortage claim (paid for 2, got 1) combined with an explicit '#fraud' accusation",
    "fraud accusation + concrete shortage claim needs human review")
add(196, "funko pops", "order_delivery", False,
    "general packaging/damage-pattern commentary, not a specific personal claim", "acknowledge feedback")
add(197, "3 late deliveries in 1 week", "order_delivery", True,
    "specific '3 in 1 week' repeated-failure count stated", "documented short-timeframe repeat pattern")
add(198, "restock", "unclear_other", False,
    "vague restock question, no specific product named", "ask which product they mean")
add(199, "poor kid is going to accidentally activate", "product_inquiry", True,
    "safety-adjacent concern about accidental activation, possibly involving a child",
    "safety concerns need specialist review regardless of normally-low-risk category")
add(200, "atv825vand we will refend", "returns_refund", True,
    "confusing multi-cancellation history combined with an explicit refund promise made by phone",
    "needs human to verify what was actually promised and follow through")

print(f"batch 5 loaded: {len(LABELS)} labels")


def main() -> None:
    df = pd.read_csv(config.PROCESSED_DIR / "golden_sample_to_label.csv").set_index("row_id")

    missing = set(df.index) - set(LABELS.keys())
    extra = set(LABELS.keys()) - set(df.index)
    if missing or extra:
        raise SystemExit(f"Label set mismatch. Missing: {missing}, extra: {extra}")

    bad = []
    for row_id, (check, intent, _esc, _reason, _note) in LABELS.items():
        text = df.loc[row_id, "customer_text"]
        if check.lower() not in text.lower():
            bad.append((row_id, check, text[:80]))
        if intent not in config.INTENT_IDS:
            bad.append((row_id, "BAD_INTENT", intent))
    if bad:
        raise SystemExit(f"Verification failed for {len(bad)} rows: {bad[:10]}")

    rows = []
    for row_id in sorted(LABELS.keys()):
        _check, intent, escalate, reason, note = LABELS[row_id]
        src = df.loc[row_id]
        rows.append({
            "id": row_id,
            "root_tweet_id": src["root_tweet_id"],
            "customer_text": src["customer_text"],
            "brand_historical_reply": src["brand_reply"],
            "weak_intent_guess": src["weak_intent"],
            "gold_intent": intent,
            "gold_escalate": escalate,
            "gold_escalation_reason": reason,
            "notes": note,
        })

    out = pd.DataFrame(rows)
    config.GOLDEN_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.GOLDEN_SET_PATH, index=False)

    print(f"\nWrote {len(out)} gold-labeled examples to {config.GOLDEN_SET_PATH}")
    print("\ngold_intent distribution:")
    print(out["gold_intent"].value_counts())
    print(f"\ngold_escalate rate: {out['gold_escalate'].mean():.1%}")
    print("\nescalate rate by intent:")
    print(out.groupby("gold_intent")["gold_escalate"].mean().sort_values(ascending=False))
    agree = (out["gold_intent"] == out["weak_intent_guess"]).mean()
    print(f"\ngold_intent vs weak_intent_guess agreement: {agree:.1%} "
          f"(expected to be well under 100% -- weak labels are noisy by design)")


if __name__ == "__main__":
    main()
