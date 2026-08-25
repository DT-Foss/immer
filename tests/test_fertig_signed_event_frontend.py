from __future__ import annotations

from fractions import Fraction
import unittest

from immer.cognition.fertig.arithmetic_ir import SolveStatus, solve
from immer.cognition.fertig.signed_event_frontend import (
    FrontendStatus,
    compile_signed_events,
    lex,
)
from immer.cognition.fertig.structural import parse_structural_problem


CASES = {
    8: (
        "It costs $194 per meter to repave a street. Monica's street is 150 "
        "meters long. How much more does it cost to repave Lewis' street, "
        "which is 490 meters long?",
        Fraction(65960),
    ),
    10: (
        "Each pole on a road intersection has 4 street lights. If the number "
        "of poles at each intersection is 6, and the road has 4 intersections, "
        "calculate the total number of functioning street lights if 20 "
        "streetlights from the total number are not working.",
        Fraction(76),
    ),
    13: (
        "A news website publishes an average of 20 political and weather news "
        "articles every day. Its sister company publishes an average of 10 "
        "business news articles daily. Calculate the total number of articles "
        "the two websites published together in February if there are 28 days "
        "in the month.",
        Fraction(840),
    ),
    27: (
        "A tomato vendor decides to switch who he buys his tomatoes for. He "
        "sells 500 tomatoes a day. He used to buy them for $.5 each but he gets "
        "a new vendor who sells them for $.4 each. How much money does he save "
        "a week?",
        Fraction(350),
    ),
    30: (
        "A perfume company is trying to create new scents. They already have 4 "
        "vanilla scents and 8 fruity scents available and they need to decide "
        "which kind of scent to focus on. They decide to focus on whichever "
        "scent sells the most and monitor their number of sales as part of their "
        "research. By the end of the day, they sell 5 of each of the vanilla "
        "scents and 2 of each of the fruity scents available. How many more "
        "vanilla scents sold compared with the fruity scents?",
        Fraction(4),
    ),
    34: (
        'Carrie is planning the caroling schedule. The choir plans to sing "Deck '
        'the Halls" twice and "Jingle Bells" once. If "Deck the Halls" is 150 '
        'seconds long and "Jingle Bells" is 240 seconds long, how long will they '
        "be caroling?",
        Fraction(540),
    ),
}

SECOND_CASES = {
    22: (
        "Ron is fed up with the pothole in front of his house. If it doesn't "
        "get fixed, it's going to do $450 worth of damage to his car. "
        "Unfortunately, the city council refuses to fix it, and will fine Ron "
        "$120 for unauthorized road maintenance if he fixes it himself. Ron "
        "will also have to buy 3 buckets of asphalt that each cost $25. How "
        "much money does Ron save by fixing the pothole?",
        Fraction(255),
        "avoided_cost_transaction",
    ),
    25: (
        "Mark decides to buy packs of old magic cards and open them to sell. "
        "He buys 3 packs for $1800 each. He gets 1 card that is worth $4000 "
        "and another card worth $1000. There are 30 more cards worth an "
        "average of $50 each. How much money profit did he make?",
        Fraction(1100),
        "profit_contribution_ledger",
    ),
    26: (
        "Jake's family wants to compare the cost of the two different "
        "amusement parks. The first amusement park has a $26 fee for each "
        "adult and a $12 fee for each child; while the second amusement park "
        "has a $14 fee for each adult and $10 for each child. If there are 2 "
        "adults and 2 children in their family, how much will they be able to "
        "save if they choose the second amusement park over the first?",
        Fraction(28),
        "alternative_cost_savings",
    ),
    37: (
        "Twenty students are working together to raise money for a charity. "
        "Each earns the same amount. The charity raises a total of $175,000. "
        "$50,000 comes from organizations and the rest from the students. How "
        "much did each student raise?",
        Fraction(6250),
        "equal_share_residual",
    ),
    53: (
        "Ellen is on a diet. She eats two carrots, a salad, and a yogurt every "
        "day. The salad costs her $6, while the yogurt is half the price. How "
        "much does Ellen pay for one carrot every day when in total she pays "
        "$11 for her goods?",
        Fraction(1),
        "unit_cost_residual",
    ),
    61: (
        "When Sophie watches her nephew, she gets out a variety of toys for "
        "him. The bag of building blocks has 31 blocks in it. The bin of "
        "stuffed animals has 8 stuffed animals inside. The tower of stacking "
        "rings has 9 multicolored rings on it. Sophie recently bought a tube "
        "of bouncy balls, bringing her total number of toys for her nephew up "
        "to 62. How many bouncy balls came in the tube?",
        Fraction(14),
        "inventory_total_residual",
    ),
}

WAVE3_CASES = {
    1: (
        "A shop sells school supplies. One notebook is sold at $1.50 each, a "
        "pen at $0.25 each, a calculator at $12 each, and a geometry set at "
        "$10. Daniel is an engineering student, and he wants to buy five "
        "notebooks, two pens, one calculator, and one geometry set. The shop "
        "gives a 10% discount on all the purchased items. How much does Daniel "
        "have to spend on all the items he wants to buy?",
        Fraction(27),
        "discounted_purchase_ledger",
    ),
    3: (
        "A food truck only sells grilled cheeses. They source their bread for "
        "$3.00 a loaf and each loaf makes 10 sandwiches. They spend $30.00 on "
        "different cheeses and condiments per 10 sandwiches. If they sell 10 "
        "sandwiches for $7.00 each, what is their net profit?",
        Fraction(37),
        "batch_sale_profit",
    ),
    19: (
        "A certain company is in the business of selling fresh fruit. One "
        "crate of such fruit consists of 5 bananas, 12 apples, and 7 oranges. "
        "The price for such a crate depends on the price of its individual "
        "fruits. One apple costs $0.5 and one banana costs twice as much. "
        "Oranges are the most expensive and cost three times as much as a "
        "banana per piece. What would be the price for such a crate of fruit?",
        Fraction(32),
        "bundle_relative_price_dag",
    ),
    21: (
        "Aiden and 12 of his friends are going to see a film at the cinema, "
        "and meet up with 7 more friends there. They each save a seat and then "
        "buy enough drinks and snacks to fill the seats. Each seat has enough "
        "room to hold one person, two drinks, and three snacks. If drinks and "
        "snacks cost $2 each, how much money, in dollars, has the group spent "
        "overall on snacks and drinks?",
        Fraction(200),
        "group_seat_purchase",
    ),
    44: (
        "Erika is saving for a new laptop. The laptop she wants costs $600. "
        "The sales assistant told her that if she traded in her old laptop, "
        "the price of the new one would be reduced by $200. She thinks this is "
        "a good deal and agrees to do it. She already has some savings in her "
        "purse, and has also been paid $150 this week for her part-time job. "
        "Her mom agrees to give her $80 to help her. If Erika now only needs "
        "an extra $50 to buy the laptop, how much money does she have in her "
        "purse?",
        Fraction(120),
        "funding_balance_residual",
    ),
    63: (
        "Zoey and Sydney are having a watermelon seed spitting contest. "
        "Whoever spits their seeds the most total distance wins. They each get "
        "one watermelon. Zoey's has 40 seeds and she spits each one 10 feet. "
        "Sydney's has 35 she spits each one 12 feet. What is the average total "
        "distance spat?",
        Fraction(410),
        "mean_participant_totals",
    ),
}

AFFINE_CASES = {
    6: (
        "At the Burger Palace restaurant, there is an enormous jar containing "
        "red, blue and green jelly beans. On the outside of the jar is a note "
        'that reads, "This jar contains 1% fewer red jelly beans than blue '
        'jelly beans and 1% more green jelly beans than blue jelly beans." '
        "If the jar contains a total of 4500 jelly beans, how many more green "
        "jelly beans does it contain than red jelly beans?",
        Fraction(30),
        "balanced_percent_category_difference",
    ),
    12: (
        "In a jewelers store, the price of a gold Jewell is 4/5 times as much "
        "as the price of a diamond Jewell. The cost of a silver Jewell is $400 "
        "less than the price of gold. If a diamond Jewell is $2000, find the "
        "total price for all three jewels.",
        Fraction(4800),
        "affine_price_chain_total",
    ),
    29: (
        "Rob, Royce, and Pedro are contractors getting ready to put a new roof "
        "on three homes. If the three homes will need 250 cases of shingles, "
        "with the first house needing 1/2 of the second, and the third needing "
        "double the first. How many cases of shingles will the third house need?",
        Fraction(100),
        "ordinal_ratio_partition",
    ),
    38: (
        "Sasha and Julie are best friends playing on opposing basketball teams. "
        "The teams have two practice games scheduled. In the first game, Sasha "
        "had the home court advantage and scored 14 points. Julie scored 4 "
        "fewer points than Sasha in the same game. Sasha always struggles "
        "during away games and their second match was at Julie's home court. "
        "Sasha scored 6 fewer points in the second game than Julie's score in "
        "the first game. How many total points did Sasha score during both games?",
        Fraction(18),
        "temporal_affine_score_chain",
    ),
    43: (
        "Jada, Rory, and Kora make clay dishes to present as art for their "
        "school project. Jada makes twice as many clay dishes as Rory, while "
        "Rory makes 20 more clay dishes than Kora. If Kora made 20 dishes, how "
        "many clay dishes they all make together?",
        Fraction(140),
        "entity_affine_chain_total",
    ),
    46: (
        "Peter has twice as many socks as Jack and half times as many dishes as "
        "jack. Jack collected twice as many dishes as socks in the store. If "
        "jack collected 60 dishes, calculate the total number of socks and "
        "dishes they have together?",
        Fraction(180),
        "cross_entity_property_dag",
    ),
    48: (
        "Claire earns 1 girl scout badge every month. It takes Amber twice as "
        "long to earn a badge than Claire. Wendy earns three times the amount "
        "of badges as Claire in the same time frame. How many more badges does "
        "Wendy earn compared to Amber in a 1 year time frame?",
        Fraction(30),
        "inverse_rate_time_difference",
    ),
    49: (
        "A bumper car rink has 12 red cars. They have 2 fewer green cars than "
        "they have red cars. They have 3 times the number of blue cars as they "
        "have green cars. The rink also has yellow cars. If the rink has 75 "
        "cars in total how many yellow cars do they have?",
        Fraction(23),
        "chained_inventory_residual",
    ),
}

PART_CAPACITY_CASES = {
    15: (
        "Ariadne has a shop selling hats of two different colors, red and green. "
        "Her sales from red hats were $400 in a particular month, half the total "
        "amount she earned from selling green hats. Calculate the total amount "
        "she made in two months if in the second month her sales were 3/4 of "
        "the total sales of the first month.",
        Fraction(2100),
        "part_scaled_period_total",
    ),
    20: (
        "Tomorrow, 42 adults and 15 babies will be attending a function at "
        "Mia’s restaurant. The restaurant has 5 times as many regular chairs "
        "as high chairs. If there are 8 high chairs, how many more chairs does "
        "she have to get?",
        Fraction(9),
        "typed_chair_capacity_deficit",
    ),
    23: (
        "Bryce and four of his friends each ordered their own pizzas after "
        "football practice. Each pizza had 12 slices. Bryce and two friends "
        "ate 2/3 of their pizzas. The two remaining friends ate ¾ of their "
        "pizzas. How many slices of pizza were left?",
        Fraction(18),
        "fractional_group_consumption_remainder",
    ),
    24: (
        "Each sleeve of graham crackers makes the base for 8 large smores. "
        "There are 3 sleeves in a box. If 9 kids want 2 smores apiece and 6 "
        "adults will eat 1 smore apiece, how many boxes of graham crackers "
        "will they need?",
        Fraction(1),
        "exact_packaging_capacity",
    ),
    28: (
        "Lorraine and Colleen are trading stickers for buttons. Each large "
        "sticker is worth a large button or three small buttons. A small "
        "sticker is worth one small button. A large button is worth three "
        "small stickers. Lorraine starts with 30 small stickers and 40 large "
        "stickers. She trades 90% of her small stickers for large buttons. She "
        "trades 50% of her large stickers for large buttons and trades the "
        "rest of them for small buttons. How many buttons does she have by "
        "the end?",
        Fraction(89),
        "typed_percentage_trade_transitions",
    ),
    45: (
        "Patrick has three glue sticks that are partially used. One has 1/6 "
        "left, the second has 2/3 left and the third one has 1/2 left. If a "
        "glue stick is 12 millimeters long originally, what is the total "
        "length of the glue sticks that are not used?",
        Fraction(16),
        "fractional_remnant_total",
    ),
    50: (
        "A three-ounce box of flavored jello makes 10 small jello cups. Greg "
        "wants to make small jello cups for his son's outdoor birthday party. "
        "There will be 30 kids and he wants to have enough so that each kid "
        "can have 4 jello cups. Jello is currently on sale for $1.25. How much "
        "will he spend on jello?",
        Fraction(15),
        "exact_package_demand_cost",
    ),
    52: (
        "Elaina is holding the final concert in her tour. To celebrate her "
        "final concert, she makes the concert twice as long as her usual "
        "concerts. At the end of the concert, she also performs a 15-minute "
        "encore. If the runtime of this final concert is 65 minutes then how "
        "long, in minutes, do her usual concerts run for?",
        Fraction(25),
        "reverse_affine_state_duration",
    ),
    57: (
        "Calvin is making soup for his family for dinner. He has a pot with "
        "enough soup to fill four adult's bowls or eight child's bowls. He is "
        "an adult and will be eating with his adult wife and their two "
        "children. If everyone eats one bowl at a meal, how many times will "
        "each child be able to have a bowl of soup for lunch from the leftover "
        "soup?",
        Fraction(1),
        "typed_bowl_capacity_leftover",
    ),
    58: (
        "Benny threw bologna at his balloons. He threw two pieces of bologna "
        "at each red balloon and three pieces of bologna at each yellow "
        "balloon. If Benny threw 58 pieces of bologna at a bundle of red and "
        "yellow balloons, and twenty of the balloons were red, then how many "
        "of the balloons in the bundle were yellow?",
        Fraction(6),
        "weighted_bundle_residual_count",
    ),
    59: (
        "Julia and Nadine were given the same amount of allowance by their "
        "mother. The two girls decided to combine their allowance to surprise "
        "their father on his birthday. They bought a cake which costs $11. "
        "They also bought 1 dozen balloons which were sold for $0.5 for 2 "
        "balloons. The remaining money was used to buy 2 tubs of ice cream for "
        "$7 each. How much did Julia and Nadine's mother give each one of them?",
        Fraction(14),
        "equal_allowance_purchase_balance",
    ),
    60: (
        "A three-toed sloth moves very slowly, and only eats when he is up in "
        "his tree. For a meal of berries, it takes the sloth 4 hours to make "
        "the trip down the tree, pick up berries, and climb back up into his "
        "tree. Assuming he picks the same number of berries on each trip, what "
        "is the least number of berries he can pick up per trip down to the "
        "ground if he wants to collect 24 berries in 8 hours?",
        Fraction(12),
        "exact_trip_capacity_minimum",
    ),
}

CALENDAR_CASE = (
    "A Reddit group has 1000 members. If each member posts an average of 3 "
    "posts per day, what's the total number of posts that the group will have "
    "in March?"
)

TEMPORAL_BLOCK_CASE = (
    "Christina records her mood every day on a calendar. Over the past thirty "
    "days of moods, she had twelve good days and eight bad days and the rest "
    "were neutral. Her first eight days were good, her second eight days were "
    "bad, and her third eight days were neutral. If the next three days were "
    "good, neutral, and good, how many good days were left in the month?"
)

ABSOLUTE_SCORE_CASE = (
    "Ava and Emma want to know who is better at the new video game Ava got for "
    "her birthday. They are each going to play one level and whoever has the "
    "highest score wins. They receive 10 points for every enemy they jump on, "
    "5 points for each berry they collect, and 30 points for every second left "
    "on the timer when they finish the level. If Ava jumps on 8 more enemies "
    "than Emma and collects 3 more berries, but finishes the level 4 seconds "
    "slower, what is the difference between their two scores?"
)

EXHAUSTIVE_SALE_CASE = (
    "Jen is planning to sell her root crops. She has 6 yams which can be sold "
    "at $1.5 each, 10 sweet potatoes that cost $2 each, and 4 carrots which "
    "cost $1.25 each. If she sells everything, how much will she earn?"
)

COMMISSIONED_LEDGER_CASE = (
    "John is a carpenter. For his friend Ali, he manufactured 4 wooden tables "
    "for $20 each and 2 roof frames for $10 each. How much does Ali have to "
    "pay John?"
)

RECURRING_RATE_CASE = (
    "Alicia's clothes have to be sent to the dry cleaners weekly. Her weekly "
    "drop-off includes 5 blouses, 2 pants and 1 skirt. If they charge her "
    "$5.00 per blouse, $6.00 per skirt and $8.00 per pair of pants, how much "
    "does she spend on dry-cleaning in 5 weeks?"
)


class SignedEventFrontendTests(unittest.TestCase):
    def test_six_shared_event_families_compile_and_solve_exactly(self) -> None:
        for offset, (question, expected) in CASES.items():
            with self.subTest(offset=offset):
                result = compile_signed_events(question)
                self.assertTrue(result.ok, result.reason)
                assert result.compiled is not None
                self.assertEqual(result.compiled.solution.target_value, expected)
                self.assertTrue(result.compiled.certificate.verified)
                numeric_tokens = [
                    token
                    for token in lex(question)
                    if token.number is not None
                    and token.text[0].isdigit()
                    or token.text.lstrip().startswith("$")
                ]
                self.assertGreaterEqual(
                    len(result.compiled.evidence_projection), len(numeric_tokens)
                )

    def test_structural_entrypoint_accepts_only_complete_signed_programs(self) -> None:
        for offset, (question, expected) in CASES.items():
            with self.subTest(offset=offset):
                parsed = parse_structural_problem(question)
                self.assertTrue(parsed.ok, parsed.reason)
                assert parsed.problem is not None
                solution = solve(parsed.problem)
                self.assertIs(solution.status, SolveStatus.UNIQUE)
                self.assertEqual(solution.target_value, expected)

    def test_domain_renames_preserve_operator_grammar(self) -> None:
        street = CASES[8][0]
        renamed = (
            street.replace("Monica", "Aria")
            .replace("Lewis", "Bram")
            .replace("street", "canal")
            .replace("repave", "restore")
        )
        result = compile_signed_events(renamed)
        self.assertTrue(result.ok, result.reason)
        assert result.compiled is not None
        self.assertEqual(result.compiled.solution.target_value, CASES[8][1])

        categories = (
            CASES[30][0]
            .replace("vanilla", "floral")
            .replace("fruity", "citrus")
            .replace("perfume", "soap")
        )
        result = compile_signed_events(categories)
        self.assertTrue(result.ok, result.reason)
        assert result.compiled is not None
        self.assertEqual(result.compiled.solution.target_value, CASES[30][1])

    def test_operator_and_state_mutations_fail_closed(self) -> None:
        mutations = (
            CASES[8][0].replace("per meter", "for material"),
            CASES[10][0].replace("are not working", "are working"),
            CASES[13][0].replace("published together", "published separately"),
            CASES[27][0].replace("a week", "eventually"),
            CASES[30][0].replace("5 of each", "5 near"),
            CASES[34][0].replace("twice", "often"),
        )
        for question in mutations:
            with self.subTest(question=question):
                self.assertFalse(compile_signed_events(question).ok)

    def test_extra_or_duplicated_numeric_surfaces_fail_closed(self) -> None:
        for question in (
            CASES[8][0] + " Reference 99.",
            CASES[8][0].replace(
                "Monica's street is 150 meters long.",
                "Monica's street is 150 meters long and half the crew rested.",
            ),
            CASES[10][0].replace(
                "calculate the total number",
                "2 supervisors are present, calculate the total number",
            ),
            CASES[10][0].replace(
                "calculate the total number",
                "each supervisor checks 2 gauges, calculate the total number",
            ),
            CASES[27][0].replace("500 tomatoes", "500 tomatoes and 500 tomatoes"),
        ):
            with self.subTest(question=question):
                result = compile_signed_events(question)
                self.assertFalse(result.ok)
                self.assertIn(
                    result.status,
                    {FrontendStatus.UNSUPPORTED, FrontendStatus.INVALID},
                )

    def test_reordered_duration_bindings_preserve_item_identity(self) -> None:
        question = CASES[34][0].replace(
            'If "Deck the Halls" is 150 seconds long and "Jingle Bells" is 240',
            'If "Jingle Bells" is 240 seconds long and "Deck the Halls" is 150',
        )
        result = compile_signed_events(question)
        self.assertTrue(result.ok, result.reason)
        assert result.compiled is not None
        self.assertEqual(result.compiled.solution.target_value, Fraction(540))

    def test_calendar_daily_rate_compiles_with_exact_month_basis(self) -> None:
        for month, days in (("March", 31), ("April", 30), ("December", 31)):
            with self.subTest(month=month):
                question = CALENDAR_CASE.replace("March", month)
                result = compile_signed_events(question)
                self.assertTrue(result.ok, result.reason)
                self.assertEqual(result.family, "calendar_daily_total")
                assert result.compiled is not None
                self.assertEqual(
                    result.compiled.solution.target_value,
                    Fraction(1000 * 3 * days),
                )
                self.assertTrue(result.compiled.certificate.verified)

                parsed = parse_structural_problem(question)
                self.assertTrue(parsed.ok, parsed.reason)
                assert parsed.problem is not None
                self.assertEqual(solve(parsed.problem).target_value, 1000 * 3 * days)

    def test_calendar_daily_rate_rejects_unproven_or_noisy_bindings(self) -> None:
        mutations = (
            CALENDAR_CASE.replace("March", "February"),
            CALENDAR_CASE.replace("March", "March or April"),
            CALENDAR_CASE.replace("each member", "each moderator"),
            CALENDAR_CASE.replace(
                "has 1000 members", "has 1000 members and 5 moderators"
            ),
            CALENDAR_CASE.replace("in March", "in an unspecified month"),
        )
        for question in mutations:
            with self.subTest(question=question):
                self.assertFalse(compile_signed_events(question).ok)

    def test_ambiguous_comparison_orientation_is_rejected(self) -> None:
        question = CASES[8][0].replace("Lewis' street", "the other street")
        result = compile_signed_events(question)
        self.assertFalse(result.ok)
        self.assertIs(result.status, FrontendStatus.AMBIGUOUS)

    def test_unrelated_and_deferred_families_do_not_match(self) -> None:
        deferred = (
            "Sarah has a 20 meter rope and a new rope costs $1.5 a meter. How "
            "much money remains after buying an unstated length?",
            "Tasha earned $80 and mowed three lawns. How much came from lemonade?",
            "What is the capital of France?",
        )
        for question in deferred:
            with self.subTest(question=question):
                result = compile_signed_events(question)
                self.assertFalse(result.ok)
                self.assertFalse(result.family_matched)


class SignedEventContributionFrontendTests(unittest.TestCase):
    def test_second_six_families_compile_with_exact_certificates(self) -> None:
        for offset, (question, expected, family) in SECOND_CASES.items():
            with self.subTest(offset=offset):
                result = compile_signed_events(question)
                self.assertTrue(result.ok, result.reason)
                self.assertEqual(result.family, family)
                assert result.compiled is not None
                self.assertIs(result.compiled.solution.status, SolveStatus.UNIQUE)
                self.assertEqual(result.compiled.solution.target_value, expected)
                self.assertTrue(result.compiled.certificate.verified)
                projection = result.compiled.evidence_projection
                self.assertTrue(projection)
                self.assertEqual(
                    len({row.evidence.evidence_id for row in projection}),
                    len(projection),
                )
                for row in projection:
                    span = row.evidence.span
                    self.assertEqual(span.source, question)
                    self.assertTrue(question[span.start : span.end])

    def test_second_six_support_domain_and_owner_renames(self) -> None:
        variants = (
            SECOND_CASES[22][0]
            .replace("Ron", "Miro")
            .replace("pothole", "culvert")
            .replace("asphalt", "gravel"),
            SECOND_CASES[25][0]
            .replace("Mark", "Niko")
            .replace("packs", "bundles")
            .replace("pack", "bundle")
            .replace("cards", "stamps")
            .replace("card", "stamp"),
            SECOND_CASES[53][0]
            .replace("Ellen", "Mara")
            .replace("carrots", "radishes")
            .replace("carrot", "radish")
            .replace("salad", "soup")
            .replace("yogurt", "juice"),
            SECOND_CASES[61][0]
            .replace("Sophie", "Talia")
            .replace("toys", "trinkets")
            .replace("blocks", "tiles")
            .replace("balls", "cubes"),
        )
        expected = (Fraction(255), Fraction(1100), Fraction(1), Fraction(14))
        for question, answer in zip(variants, expected, strict=True):
            with self.subTest(question=question):
                result = compile_signed_events(question)
                self.assertTrue(result.ok, result.reason)
                assert result.compiled is not None
                self.assertEqual(result.compiled.solution.target_value, answer)

    def test_declarative_reordering_preserves_bound_ledgers(self) -> None:
        variants = (
            (
                "Ron is fed up with the pothole in front of his house. Ron will "
                "also have to buy 3 buckets of asphalt that each cost $25. "
                "Unfortunately, the city council refuses to fix it, and will "
                "fine Ron $120 for unauthorized road maintenance if he fixes it "
                "himself. If it doesn't get fixed, it's going to do $450 worth "
                "of damage to his car. How much money does Ron save by fixing "
                "the pothole?"
            ),
            (
                "Mark decides to buy packs of old magic cards and open them to "
                "sell. There are 30 more cards worth an average of $50 each. He "
                "gets 1 card that is worth $4000 and another card worth $1000. "
                "He buys 3 packs for $1800 each. How much money profit did he make?"
            ),
            (
                "Twenty students are working together to raise money for a "
                "charity. $50,000 comes from organizations and the rest from the "
                "students. The charity raises a total of $175,000. Each earns "
                "the same amount. How much did each student raise?"
            ),
            (
                "When Sophie watches her nephew, she gets out a variety of toys "
                "for him. The tower of stacking rings has 9 multicolored rings "
                "on it. The bag of building blocks has 31 blocks in it. The bin "
                "of stuffed animals has 8 stuffed animals inside. Sophie "
                "recently bought a tube of bouncy balls, bringing her total "
                "number of toys for her nephew up to 62. How many bouncy balls "
                "came in the tube?"
            ),
        )
        expected = (Fraction(255), Fraction(1100), Fraction(6250), Fraction(14))
        for question, answer in zip(variants, expected, strict=True):
            with self.subTest(question=question):
                result = compile_signed_events(question)
                self.assertTrue(result.ok, result.reason)
                assert result.compiled is not None
                self.assertEqual(result.compiled.solution.target_value, answer)

    def test_operator_direction_and_scope_mutations_abstain(self) -> None:
        mutations = (
            SECOND_CASES[22][0].replace("that each cost", "that cost"),
            SECOND_CASES[22][0].replace("fixing the pothole", "fixing the roof"),
            SECOND_CASES[25][0].replace("30 more cards", "30 more coins"),
            SECOND_CASES[25][0].replace("$1800 each", "$1800 total"),
            SECOND_CASES[26][0].replace("2 children", "2 teachers"),
            SECOND_CASES[26][0].replace(
                "choose the second amusement park over the first",
                "choose the first amusement park over the second",
            ),
            SECOND_CASES[37][0].replace(
                "Each earns the same", "Each earns a different"
            ),
            SECOND_CASES[37][0].replace("each student raise", "each teacher raise"),
            SECOND_CASES[53][0].replace("half the price", "double the price"),
            SECOND_CASES[53][0].replace("two carrots", "two apples"),
            SECOND_CASES[61][0].replace(
                "total number of toys", "total number of tools"
            ),
            SECOND_CASES[61][0].replace(
                "How many bouncy balls", "How many glass beads"
            ),
            SECOND_CASES[61][0].replace("has 8 stuffed animals", "has 8 wooden boxes"),
        )
        for question in mutations:
            with self.subTest(question=question):
                self.assertFalse(compile_signed_events(question).ok)

    def test_unrelated_numeric_surfaces_fail_evidence_closure(self) -> None:
        for offset, (question, _, _) in SECOND_CASES.items():
            with self.subTest(offset=offset):
                noisy = question.replace("?", " with reference number 99?")
                result = compile_signed_events(noisy)
                self.assertFalse(result.ok)
                self.assertIn(
                    result.status,
                    {FrontendStatus.UNSUPPORTED, FrontendStatus.INVALID},
                )

    def test_cross_scope_contributions_never_enter_the_ledger(self) -> None:
        repair = (
            SECOND_CASES[22][0]
            .replace(
                "will fine Ron $120 for unauthorized road maintenance if he fixes it himself",
                "will fine Ron $120 for parking in the wrong place",
            )
            .replace(
                "3 buckets of asphalt that each cost $25",
                "3 buckets of asphalt that each cost $25 for his garden path",
            )
        )
        foreign_actor = (
            SECOND_CASES[25][0]
            .replace("He gets 1 card", "Luke gets 1 card")
            .replace("did he make", "did Mark make")
        )
        mixed_actor_pronouns = SECOND_CASES[25][0].replace(
            "He gets 1 card", "She gets 1 card"
        )
        foreign_damage = SECOND_CASES[22][0].replace(
            "If it doesn't get fixed", "If his roof doesn't get fixed"
        )
        foreign_damage_car = SECOND_CASES[22][0].replace(
            "If it doesn't get fixed", "If the car doesn't get fixed"
        )
        foreign_inventory = SECOND_CASES[61][0].replace(
            "Sophie recently bought",
            "The shelf of toy cars has 2 cars on it. Sophie recently bought",
        )
        foreign_share = SECOND_CASES[37][0].replace(
            "How much did each student raise?",
            "Another club says each earns the same amount too. How much did "
            "each student raise?",
        )
        foreign_price = SECOND_CASES[53][0].replace(
            "The salad costs her", "Ben says the salad costs her"
        )
        foreign_price_anchor = SECOND_CASES[53][0].replace(
            "half the price", "half the smoothie price"
        )
        foreign_unit_cost_owner = SECOND_CASES[53][0].replace(
            "Ellen is on a diet", "Mara is on a diet"
        )
        foreign_total_owner = SECOND_CASES[61][0].replace(
            "Sophie recently bought", "Mia recently bought"
        )
        foreign_total_possessive = SECOND_CASES[61][0].replace(
            "bringing her total", "bringing Mia's total"
        )
        for question in (
            repair,
            foreign_actor,
            mixed_actor_pronouns,
            foreign_damage,
            foreign_damage_car,
            foreign_inventory,
            foreign_share,
            foreign_price,
            foreign_price_anchor,
            foreign_unit_cost_owner,
            foreign_total_owner,
            foreign_total_possessive,
        ):
            with self.subTest(question=question):
                self.assertFalse(compile_signed_events(question).ok)

    def test_two_and_twenty_are_numeric_only_inside_bound_productions(self) -> None:
        unrelated = lex("Two notes and twenty labels are on a desk.")
        self.assertTrue(
            all(
                token.number is None
                for token in unrelated
                if token.norm in {"two", "twenty"}
            )
        )
        for offset, surface, value in ((37, "Twenty", 20), (53, "two", 2)):
            result = compile_signed_events(SECOND_CASES[offset][0])
            self.assertTrue(result.ok, result.reason)
            assert result.compiled is not None
            matches = [
                row.evidence
                for row in result.compiled.evidence_projection
                if row.evidence.span.source[
                    row.evidence.span.start : row.evidence.span.end
                ]
                == surface
            ]
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0].value, value)


class SignedEventAggregationWaveTests(unittest.TestCase):
    def test_safe_wave_compiles_independently_derived_exact_values(self) -> None:
        # 1: basket 30 less 10%; 3: sales 70 less batch costs 33;
        # 19: 5*1 + 12*(1/2) + 7*3; 21: 20 seats * 5 items * $2;
        # 44: 600-200-150-80-50; 63: mean(40*10, 35*12).
        for offset, (question, expected, family) in WAVE3_CASES.items():
            with self.subTest(offset=offset):
                result = compile_signed_events(question)
                self.assertTrue(result.ok, result.reason)
                self.assertEqual(result.family, family)
                assert result.compiled is not None
                self.assertIs(result.compiled.solution.status, SolveStatus.UNIQUE)
                self.assertEqual(result.compiled.solution.target_value, expected)
                self.assertTrue(result.compiled.certificate.verified)
                self.assertEqual(
                    len(result.compiled.evidence_projection),
                    len(
                        {
                            row.evidence.evidence_id
                            for row in result.compiled.evidence_projection
                        }
                    ),
                )
                parsed = parse_structural_problem(question)
                self.assertTrue(parsed.ok, parsed.reason)
                assert parsed.problem is not None
                self.assertEqual(solve(parsed.problem).target_value, expected)

    def test_domain_and_actor_renames_preserve_relations(self) -> None:
        variants = (
            WAVE3_CASES[1][0]
            .replace("Daniel", "Mira")
            .replace("notebook", "journal")
            .replace("a pen at", "a marker at")
            .replace("two pens", "two markers")
            .replace("calculator", "compass")
            .replace("geometry set", "drafting kit"),
            WAVE3_CASES[19][0]
            .replace("crate", "basket")
            .replace("bananas", "plums")
            .replace("banana", "plum")
            .replace("apples", "pears")
            .replace("apple", "pear")
            .replace("oranges", "melons")
            .replace("Oranges", "Melons"),
            WAVE3_CASES[44][0].replace("Erika", "Lina").replace("laptop", "bicycle"),
            WAVE3_CASES[63][0].replace("Zoey", "Mira").replace("Sydney", "Talia"),
        )
        expected = (Fraction(27), Fraction(32), Fraction(120), Fraction(410))
        for question, answer in zip(variants, expected, strict=True):
            with self.subTest(question=question):
                result = compile_signed_events(question)
                self.assertTrue(result.ok, result.reason)
                assert result.compiled is not None
                self.assertEqual(result.compiled.solution.target_value, answer)

    def test_relation_unit_actor_and_time_mutations_abstain(self) -> None:
        mutations = (
            WAVE3_CASES[1][0].replace(
                "on all the purchased items", "on notebooks only"
            ),
            WAVE3_CASES[1][0].replace("one calculator", "one ruler"),
            WAVE3_CASES[3][0].replace("per 10 sandwiches", "per 10 salads"),
            WAVE3_CASES[3][0].replace("They spend $30.00", "Mara spends $30.00"),
            WAVE3_CASES[3][0].replace(
                "per 10 sandwiches.", "per 10 sandwiches per day."
            ),
            WAVE3_CASES[19][0].replace("twice as much", "twice as many"),
            WAVE3_CASES[19][0].replace(
                "as a banana per piece", "as an apple per piece"
            ),
            WAVE3_CASES[21][0].replace("save a seat", "save a table"),
            WAVE3_CASES[21][0].replace("If drinks and snacks cost", "If drinks cost"),
            WAVE3_CASES[44][0].replace("paid $150 this week", "paid $150 last week"),
            WAVE3_CASES[44][0].replace("She already has", "Ben already has"),
            WAVE3_CASES[63][0].replace("12 feet", "12 yards"),
            WAVE3_CASES[63][0].replace(
                "average total distance", "average seed distance"
            ),
        )
        for question in mutations:
            with self.subTest(question=question):
                self.assertFalse(compile_signed_events(question).ok)


class SignedEventScopedAffineWaveTests(unittest.TestCase):
    def test_eight_scoped_affine_dags_are_exact_and_certified(self) -> None:
        for offset, (question, expected, family) in AFFINE_CASES.items():
            with self.subTest(offset=offset):
                result = compile_signed_events(question)
                self.assertTrue(result.ok, result.reason)
                self.assertEqual(result.family, family)
                assert result.compiled is not None
                self.assertIs(result.compiled.solution.status, SolveStatus.UNIQUE)
                self.assertEqual(result.compiled.solution.target_value, expected)
                self.assertTrue(result.compiled.certificate.verified)
                self.assertEqual(
                    len(result.compiled.evidence_projection),
                    len(
                        {
                            row.evidence.evidence_id
                            for row in result.compiled.evidence_projection
                        }
                    ),
                )
                parsed = parse_structural_problem(question)
                self.assertTrue(parsed.ok, parsed.reason)
                assert parsed.problem is not None
                self.assertEqual(solve(parsed.problem).target_value, expected)

    def test_fraction_percent_and_year_scaling_keep_exact_units(self) -> None:
        percent = compile_signed_events(AFFINE_CASES[6][0])
        self.assertTrue(percent.ok, percent.reason)
        assert percent.compiled is not None
        percent_rows = [
            row.evidence
            for row in percent.compiled.evidence_projection
            if row.evidence.unit.symbol == "%"
        ]
        self.assertEqual([row.value for row in percent_rows], [1, 1])
        self.assertTrue(all(row.unit.scale == Fraction(1, 100) for row in percent_rows))
        self.assertEqual(
            percent.compiled.problem.target.unit.dimensions, (("count", 1),)
        )

        ratio = compile_signed_events(AFFINE_CASES[12][0])
        self.assertTrue(ratio.ok, ratio.reason)
        assert ratio.compiled is not None
        fraction_rows = [
            row.evidence
            for row in ratio.compiled.evidence_projection
            if row.evidence.value == Fraction(4, 5)
        ]
        self.assertEqual(len(fraction_rows), 1)
        span = fraction_rows[0].span
        self.assertEqual(AFFINE_CASES[12][0][span.start : span.end], "4/5")

        annual = compile_signed_events(AFFINE_CASES[48][0])
        self.assertTrue(annual.ok, annual.reason)
        assert annual.compiled is not None
        self.assertEqual(
            annual.compiled.problem.target.unit.dimensions, (("count", 1),)
        )
        self.assertEqual(annual.compiled.problem.target.unit.scale, 1)

    def test_owner_category_and_item_renames_preserve_affine_grammar(self) -> None:
        variants = (
            AFFINE_CASES[6][0]
            .replace("red", "amber")
            .replace("blue", "cyan")
            .replace("green", "violet"),
            AFFINE_CASES[12][0]
            .replace("gold", "ruby")
            .replace("diamond", "opal")
            .replace("silver", "pearl")
            .replace("Jewell", "Gem")
            .replace("jewels", "gems"),
            AFFINE_CASES[38][0].replace("Sasha", "Mira").replace("Julie", "Talia"),
            AFFINE_CASES[43][0]
            .replace("Jada", "Mira")
            .replace("Rory", "Talia")
            .replace("Kora", "Nia"),
            AFFINE_CASES[49][0]
            .replace("red", "amber")
            .replace("green", "cyan")
            .replace("blue", "violet")
            .replace("yellow", "white"),
        )
        expected = (30, 4800, 18, 140, 23)
        for question, answer in zip(variants, expected, strict=True):
            with self.subTest(question=question):
                result = compile_signed_events(question)
                self.assertTrue(result.ok, result.reason)
                assert result.compiled is not None
                self.assertEqual(result.compiled.solution.target_value, answer)

    def test_direction_state_item_and_time_mutations_abstain(self) -> None:
        mutations = (
            AFFINE_CASES[6][0].replace(
                "more green jelly beans than blue", "more green jelly beans than red"
            ),
            AFFINE_CASES[6][0].replace("1% more green", "2% more green"),
            AFFINE_CASES[6][0].replace(
                "red, blue and green", "red, blue, yellow and green"
            ),
            AFFINE_CASES[12][0].replace(
                "less than the price of gold", "less than the price of diamond"
            ),
            AFFINE_CASES[12][0].replace("4/5 times as much", "4/5 more than"),
            AFFINE_CASES[29][0].replace("double the first", "double the second"),
            AFFINE_CASES[29][0].replace("250 cases of shingles", "250 cases of tiles"),
            AFFINE_CASES[38][0].replace("in the same game", "in the second game"),
            AFFINE_CASES[38][0].replace(
                "Julie's score in the first game", "Julie's score in the second game"
            ),
            AFFINE_CASES[43][0].replace("than Kora", "than Jada"),
            AFFINE_CASES[43][0].replace(
                "how many clay dishes", "how many clay sculptures"
            ),
            AFFINE_CASES[46][0].replace("half times as many", "half more than"),
            AFFINE_CASES[46][0].replace(
                "twice as many dishes as socks in the store",
                "twice as many socks as dishes in the store",
            ),
            AFFINE_CASES[48][0].replace("twice as long", "twice as many"),
            AFFINE_CASES[48][0].replace("compared to Amber", "compared to Claire"),
            AFFINE_CASES[49][0].replace(
                "blue cars as they have green", "blue cars as they have red"
            ),
            AFFINE_CASES[49][0].replace("2 fewer green", "2 more green"),
        )
        for question in mutations:
            with self.subTest(question=question):
                self.assertFalse(compile_signed_events(question).ok)


class SignedEventPartCapacityWaveTests(unittest.TestCase):
    def test_twelve_part_capacity_cases_are_exact_and_certified(self) -> None:
        for offset, (question, expected, family) in PART_CAPACITY_CASES.items():
            with self.subTest(offset=offset):
                result = compile_signed_events(question)
                self.assertTrue(result.ok, result.reason)
                self.assertEqual(result.family, family)
                assert result.compiled is not None
                self.assertIs(result.compiled.solution.status, SolveStatus.UNIQUE)
                self.assertEqual(result.compiled.solution.target_value, expected)
                self.assertTrue(result.compiled.certificate.verified)
                self.assertEqual(
                    len(result.compiled.evidence_projection),
                    len(
                        {
                            row.evidence.evidence_id
                            for row in result.compiled.evidence_projection
                        }
                    ),
                )
                parsed = parse_structural_problem(question)
                self.assertTrue(parsed.ok, parsed.reason)
                assert parsed.problem is not None
                self.assertEqual(solve(parsed.problem).target_value, expected)

    def test_unicode_fraction_percent_dozen_and_capacity_units_are_exact(self) -> None:
        pizza = compile_signed_events(PART_CAPACITY_CASES[23][0])
        self.assertTrue(pizza.ok, pizza.reason)
        assert pizza.compiled is not None
        unicode_rows = [
            row.evidence
            for row in pizza.compiled.evidence_projection
            if row.evidence.value == Fraction(3, 4)
        ]
        self.assertEqual(len(unicode_rows), 1)
        span = unicode_rows[0].span
        self.assertEqual(PART_CAPACITY_CASES[23][0][span.start : span.end], "¾")

        trades = compile_signed_events(PART_CAPACITY_CASES[28][0])
        self.assertTrue(trades.ok, trades.reason)
        assert trades.compiled is not None
        percent_rows = [
            row.evidence
            for row in trades.compiled.evidence_projection
            if row.evidence.unit.symbol == "%"
        ]
        self.assertEqual(sorted(row.value for row in percent_rows), [50, 50, 90])
        self.assertTrue(all(row.unit.scale == Fraction(1, 100) for row in percent_rows))

        allowance = compile_signed_events(PART_CAPACITY_CASES[59][0])
        self.assertTrue(allowance.ok, allowance.reason)
        assert allowance.compiled is not None
        dozen_rows = [
            row.evidence
            for row in allowance.compiled.evidence_projection
            if row.evidence.value == 12
            and row.evidence.span.source[
                row.evidence.span.start : row.evidence.span.end
            ]
            == "dozen"
        ]
        self.assertEqual(len(dozen_rows), 1)

        for offset in (20, 23, 24, 57, 58, 60):
            result = compile_signed_events(PART_CAPACITY_CASES[offset][0])
            self.assertTrue(result.ok, result.reason)
            assert result.compiled is not None
            self.assertEqual(
                result.compiled.problem.target.unit.dimensions, (("count", 1),)
            )

    def test_owner_renames_preserve_part_capacity_relations(self) -> None:
        variants = (
            PART_CAPACITY_CASES[15][0].replace("Ariadne", "Mara"),
            PART_CAPACITY_CASES[23][0].replace("Bryce", "Mira"),
            PART_CAPACITY_CASES[52][0].replace("Elaina", "Talia"),
            PART_CAPACITY_CASES[58][0].replace("Benny", "Miro"),
            PART_CAPACITY_CASES[59][0]
            .replace("Julia", "Mira")
            .replace("Nadine", "Talia"),
        )
        expected = (2100, 18, 25, 6, 14)
        for question, answer in zip(variants, expected, strict=True):
            with self.subTest(question=question):
                result = compile_signed_events(question)
                self.assertTrue(result.ok, result.reason)
                assert result.compiled is not None
                self.assertEqual(result.compiled.solution.target_value, answer)

    def test_part_state_item_and_distribution_mutations_abstain(self) -> None:
        mutations = (
            PART_CAPACITY_CASES[15][0].replace(
                "half the total amount", "half the number of hats"
            ),
            PART_CAPACITY_CASES[20][0].replace("15 babies", "15 toddlers"),
            PART_CAPACITY_CASES[23][0].replace(
                "two remaining friends", "three remaining friends"
            ),
            PART_CAPACITY_CASES[28][0].replace(
                "rest of them for small buttons", "rest of them for large buttons"
            ),
            PART_CAPACITY_CASES[45][0].replace("1/2 left", "1/2 used"),
            PART_CAPACITY_CASES[45][0].replace(
                "glue sticks that are not used", "candles that are not used"
            ),
            PART_CAPACITY_CASES[52][0].replace(
                "end of the concert", "start of another concert"
            ),
            PART_CAPACITY_CASES[57][0].replace("adult wife", "child wife"),
            PART_CAPACITY_CASES[58][0].replace(
                "three pieces of bologna at each yellow",
                "three pieces of cheese at each yellow",
            ),
            PART_CAPACITY_CASES[59][0].replace(
                "remaining money was used", "some other money was used"
            ),
            PART_CAPACITY_CASES[60][0].replace(
                "same number of berries", "different number of berries"
            ),
        )
        for question in mutations:
            with self.subTest(question=question):
                self.assertFalse(compile_signed_events(question).ok)

    def test_nonexact_capacity_divisions_use_explicit_ceiling(self) -> None:
        cases = (
            (PART_CAPACITY_CASES[24][0].replace("9 kids", "10 kids"), Fraction(2)),
            (
                PART_CAPACITY_CASES[50][0].replace("30 kids", "31 kids"),
                Fraction(65, 4),
            ),
            (
                PART_CAPACITY_CASES[60][0].replace("24 berries", "25 berries"),
                Fraction(13),
            ),
        )
        for question, expected in cases:
            with self.subTest(question=question):
                result = compile_signed_events(question)
                self.assertTrue(result.ok, result.reason)
                assert result.compiled is not None
                self.assertEqual(result.compiled.solution.target_value, expected)
                self.assertTrue(result.compiled.certificate.verified)

    def test_cross_scope_and_extra_evidence_fail_closed(self) -> None:
        cross_scope = (
            PART_CAPACITY_CASES[15][0].replace(
                "second month her sales", "second shop its sales"
            ),
            PART_CAPACITY_CASES[20][0].replace(
                "If there are 8 high chairs", "If another restaurant has 8 high chairs"
            ),
            PART_CAPACITY_CASES[23][0].replace(
                "Bryce and two friends ate", "Mira and two friends ate"
            ),
            PART_CAPACITY_CASES[23][0].replace(
                "slices of pizza were left", "slices of cake were left"
            ),
            PART_CAPACITY_CASES[24][0].replace(
                "3 sleeves in a box", "3 sleeves in another package"
            ),
            PART_CAPACITY_CASES[24][0].replace(
                "6 adults will eat 1 smore", "6 adults will eat 1 marshmallow"
            ),
            PART_CAPACITY_CASES[24][0].replace(
                "boxes of graham crackers", "boxes of marshmallows"
            ),
            PART_CAPACITY_CASES[28][0].replace("Lorraine starts", "Mira starts"),
            PART_CAPACITY_CASES[50][0].replace(
                "Jello is currently on sale", "Pudding is currently on sale"
            ),
            PART_CAPACITY_CASES[52][0].replace(
                "she also performs", "Mira also performs"
            ),
            PART_CAPACITY_CASES[57][0].replace(
                "their two children", "Mira's two children"
            ),
            PART_CAPACITY_CASES[57][0].replace("leftover soup", "leftover stew"),
            PART_CAPACITY_CASES[58][0].replace("If Benny threw 58", "If Mira threw 58"),
            PART_CAPACITY_CASES[58][0].replace(
                "bundle of red and yellow balloons",
                "bundle of red and blue balloons",
            ),
            PART_CAPACITY_CASES[59][0].replace(
                "Julia and Nadine's mother", "Julia and Mira's mother"
            ),
            PART_CAPACITY_CASES[60][0].replace(
                "if he wants to collect", "if another sloth wants to collect"
            ),
            PART_CAPACITY_CASES[60][0].replace("pick up berries", "pick up apples"),
            PART_CAPACITY_CASES[60][0].replace(
                "least number of berries he can pick up",
                "least number of apples he can pick up",
            ),
        )
        for question in cross_scope:
            with self.subTest(question=question):
                self.assertFalse(compile_signed_events(question).ok)
        for offset, (question, _, _) in PART_CAPACITY_CASES.items():
            with self.subTest(offset=offset):
                self.assertFalse(compile_signed_events(question + " Reference 99.").ok)

    def test_temporal_share_and_installation_ambiguities_remain_deferred(self) -> None:
        deferred = (
            "Over thirty days there were twelve good days. The first eight "
            "were good and the next three were good, neutral, good. How many "
            "good days were left?",
            "There are three puppies, five koalas, two zebras, and four frogs. "
            "How many goats make goats 30% of the final collection?",
            "Installation includes 4 mirrors, 2 shelves, 1 chandelier, and 10 "
            "pictures. Extra items cost $15. Angela has 6 mirrors, 2 "
            "chandeliers, and 20 pictures. What does installation cost?",
        )
        for question in deferred:
            with self.subTest(question=question):
                self.assertFalse(compile_signed_events(question).ok)

    def test_cross_scope_and_extra_evidence_never_enter_affine_dags(self) -> None:
        cross_scope = (
            AFFINE_CASES[6][0].replace(
                "If the jar contains a total", "If another jar contains a total"
            ),
            AFFINE_CASES[6][0].replace(
                "If the jar contains a total",
                "If the jar and another display together contain a total",
            ),
            AFFINE_CASES[12][0].replace(
                "If a diamond Jewell is $2000", "If an opal Jewell is $2000"
            ),
            AFFINE_CASES[12][0].replace(
                "If a diamond Jewell is $2000",
                "If a diamond display case costs $2000",
            ),
            AFFINE_CASES[29][0].replace(
                "How many cases of shingles", "How many cases of tiles"
            ),
            AFFINE_CASES[38][0].replace("Sasha scored 6 fewer", "Mira scored 6 fewer"),
            AFFINE_CASES[43][0].replace("while Rory makes 20", "while Mira makes 20"),
            AFFINE_CASES[43][0].replace(
                "clay dishes they all make together",
                "clay dishes do the teachers all make together",
            ),
            AFFINE_CASES[46][0].replace("Jack collected twice", "Mira collected twice"),
            AFFINE_CASES[48][0].replace(
                "in the same time frame", "in another time frame"
            ),
            AFFINE_CASES[49][0].replace("If the rink has 75", "If another rink has 75"),
        )
        for question in cross_scope:
            with self.subTest(question=question):
                self.assertFalse(compile_signed_events(question).ok)
        for offset, (question, _, _) in AFFINE_CASES.items():
            with self.subTest(offset=offset):
                self.assertFalse(compile_signed_events(question + " Reference 99.").ok)

    def test_remaining_affine_candidates_stay_deferred(self) -> None:
        deferred = (
            "Ava gets 10 points per enemy, 5 per berry, and 30 per timer "
            "second. She gets 8 more enemies and 3 more berries than Emma but "
            "finishes 4 seconds slower. What is the difference in their scores?",
            "Tasha made $80 from lemonade and mowing. She mowed one lawn three "
            "times and another five times as often as Joe, who paid $6. How "
            "much came from lemonade?",
            "Yesterday Denise read 10 pages and Daniel 13. Today Denise read 5 "
            "more than Daniel read yesterday and Daniel read none. How many "
            "more pages did Denise read than Daniel?",
        )
        for question in deferred:
            with self.subTest(question=question):
                self.assertFalse(compile_signed_events(question).ok)

    def test_cross_scope_or_extra_numeric_evidence_abstains(self) -> None:
        cross_scope = (
            WAVE3_CASES[1][0].replace(
                "The shop gives a 10% discount",
                "Another shop gives a 10% discount",
            ),
            WAVE3_CASES[1][0].replace("and he wants to buy", "and Ben wants to buy"),
            WAVE3_CASES[3][0].replace(
                "They spend $30.00", "Another truck spends $30.00"
            ),
            WAVE3_CASES[19][0].replace("7 oranges", "7 oranges and 2 pears"),
            WAVE3_CASES[21][0].replace(
                "They each save a seat", "Another group each save a seat"
            ),
            WAVE3_CASES[21][0].replace(
                "and then buy enough", "and then Ben buys enough"
            ),
            WAVE3_CASES[21][0].replace(
                "If drinks and snacks cost", "If Ben says drinks and snacks cost"
            ),
            WAVE3_CASES[44][0].replace("Her mom agrees", "Mara's mom agrees"),
            WAVE3_CASES[44][0].replace(
                "She already has some savings in her purse, and has also been paid",
                "Ben has also been paid",
            ),
            WAVE3_CASES[44][0].replace(
                "and has also been paid", "and Ben has also been paid"
            ),
            WAVE3_CASES[44][0].replace("give her $80", "give Ben $80"),
            WAVE3_CASES[44][0].replace(
                "she traded in her old laptop", "she traded in Mara's old laptop"
            ),
            WAVE3_CASES[44][0].replace(
                "her old laptop, the price of the new one",
                "her old bike, although she mentioned a laptop, the price of the new one",
            ),
            WAVE3_CASES[63][0].replace(
                "What is the average",
                "Kai's has 2 seeds and he spits each one 3 feet. What is the average",
            ),
        )
        for question in cross_scope:
            with self.subTest(question=question):
                self.assertFalse(compile_signed_events(question).ok)
        for offset, (question, _, _) in WAVE3_CASES.items():
            with self.subTest(offset=offset):
                self.assertFalse(
                    compile_signed_events(
                        question.replace("?", " with unrelated reference 99?")
                    ).ok
                )

    def test_safe_clause_reordering_preserves_the_dag(self) -> None:
        purchase = WAVE3_CASES[1][0]
        reordered = purchase.replace(
            "One notebook is sold at $1.50 each, a pen at $0.25 each, a "
            "calculator at $12 each, and a geometry set at $10. Daniel is an "
            "engineering student, and he wants to buy five notebooks, two "
            "pens, one calculator, and one geometry set. The shop gives a 10% "
            "discount on all the purchased items.",
            "The shop gives a 10% discount on all the purchased items. Daniel "
            "is an engineering student, and he wants to buy five notebooks, "
            "two pens, one calculator, and one geometry set. One notebook is "
            "sold at $1.50 each, a pen at $0.25 each, a calculator at $12 each, "
            "and a geometry set at $10.",
        )
        result = compile_signed_events(reordered)
        self.assertTrue(result.ok, result.reason)
        assert result.compiled is not None
        self.assertEqual(result.compiled.solution.target_value, 27)

    def test_percentage_evidence_preserves_surface_value_and_unit_scale(self) -> None:
        question = WAVE3_CASES[1][0]
        result = compile_signed_events(question)
        self.assertTrue(result.ok, result.reason)
        assert result.compiled is not None
        percentages = [
            row.evidence
            for row in result.compiled.evidence_projection
            if row.evidence.unit.symbol == "%"
        ]
        self.assertEqual(len(percentages), 1)
        self.assertEqual(percentages[0].value, 10)
        self.assertEqual(percentages[0].unit.scale, Fraction(1, 100))
        span = percentages[0].span
        self.assertEqual(question[span.start : span.end], "10")

    def test_ambiguous_or_unitless_candidates_remain_deferred(self) -> None:
        deferred = (
            "On Tuesday, Clara bought 20 pomegranates at $20 each. At the till "
            "she got $2 off because she had a voucher. The next day, the price "
            "shot to $30 per fruit, but the store also offered a 10% discount "
            "on the total cost. Sheila took advantage of the discount and "
            "bought 20 pomegranates. What is the difference between the final "
            "prices paid for the pomegranates on the two days?",
            "Britany records 18 4-minute TikTok videos each week. She spends 2 "
            "hours a week writing songs, and 15 minutes six days a week doing "
            "makeup. How much time does Britany spend on TikTok in a month with "
            "four weeks?",
            "An installation package costs $129 and includes 4 mirrors, 2 "
            "shelves, 1 chandelier, and 10 pictures. Extra items cost $15 each. "
            "Angela has 6 mirrors, 2 chandeliers, and 20 pictures. What is the "
            "cost?",
        )
        for question in deferred:
            with self.subTest(question=question):
                self.assertFalse(compile_signed_events(question).ok)


class SignedEventGroundOperatorWaveTests(unittest.TestCase):
    def test_temporal_block_and_absolute_score_delta_are_exact(self) -> None:
        cases = (
            (
                TEMPORAL_BLOCK_CASE,
                Fraction(2),
                "temporal_categorical_block_remainder",
            ),
            (
                ABSOLUTE_SCORE_CASE,
                Fraction(25),
                "absolute_weighted_score_difference",
            ),
        )
        for question, expected, family in cases:
            with self.subTest(family=family):
                result = compile_signed_events(question)
                self.assertTrue(result.ok, result.reason)
                self.assertEqual(result.family, family)
                assert result.compiled is not None
                self.assertEqual(result.compiled.solution.target_value, expected)
                self.assertTrue(result.compiled.certificate.verified)
                parsed = parse_structural_problem(question)
                self.assertTrue(parsed.ok, parsed.reason)
                assert parsed.problem is not None
                self.assertEqual(solve(parsed.problem).target_value, expected)

    def test_temporal_categories_and_score_events_can_be_renamed(self) -> None:
        temporal = (
            TEMPORAL_BLOCK_CASE.replace("Christina", "Mara")
            .replace("good", "calm")
            .replace("bad", "tense")
            .replace("neutral", "steady")
        )
        score = (
            ABSOLUTE_SCORE_CASE.replace("Ava", "Mira")
            .replace("Emma", "Talia")
            .replace("enemies", "targets")
            .replace("enemy", "target")
            .replace("berries", "coins")
            .replace("berry", "coin")
        )
        for question, expected in ((temporal, 2), (score, 25)):
            with self.subTest(question=question):
                result = compile_signed_events(question)
                self.assertTrue(result.ok, result.reason)
                assert result.compiled is not None
                self.assertEqual(result.compiled.solution.target_value, expected)

    def test_temporal_state_count_and_owner_mutations_fail_closed(self) -> None:
        mutations = (
            TEMPORAL_BLOCK_CASE.replace("next three days", "next four days"),
            TEMPORAL_BLOCK_CASE.replace(
                "third eight days were neutral", "third eight days were bad"
            ),
            TEMPORAL_BLOCK_CASE.replace("twelve good days", "seven good days"),
            TEMPORAL_BLOCK_CASE.replace("Her first eight", "Their first eight"),
            TEMPORAL_BLOCK_CASE + " Reference 99.",
        )
        for question in mutations:
            with self.subTest(question=question):
                self.assertFalse(compile_signed_events(question).ok)


class SignedEventConservativeCoreferenceWaveTests(unittest.TestCase):
    def test_three_closed_pronoun_rate_ledgers_are_exact(self) -> None:
        cases = (
            (EXHAUSTIVE_SALE_CASE, Fraction(34), "exhaustive_unit_rate_ledger"),
            (
                COMMISSIONED_LEDGER_CASE,
                Fraction(100),
                "exhaustive_unit_rate_ledger",
            ),
            (
                RECURRING_RATE_CASE,
                Fraction(235),
                "recurring_pronoun_rate_ledger",
            ),
        )
        for question, expected, family in cases:
            with self.subTest(family=family, question=question):
                result = compile_signed_events(question)
                self.assertTrue(result.ok, result.reason)
                self.assertEqual(result.family, family)
                assert result.compiled is not None
                self.assertEqual(result.compiled.solution.target_value, expected)
                self.assertTrue(result.compiled.certificate.verified)
                self.assertEqual(
                    len(result.compiled.evidence_projection),
                    len(
                        {
                            row.evidence.evidence_id
                            for row in result.compiled.evidence_projection
                        }
                    ),
                )

    def test_owner_and_item_renames_preserve_pronoun_binding(self) -> None:
        sale = (
            EXHAUSTIVE_SALE_CASE.replace("Jen", "Mara")
            .replace("yams", "melons")
            .replace("sweet potatoes", "red peppers")
            .replace("carrots", "onions")
        )
        commissioned = COMMISSIONED_LEDGER_CASE.replace("John", "Miro").replace(
            "Ali", "Talia"
        )
        recurring = (
            RECURRING_RATE_CASE.replace("Alicia", "Mara")
            .replace("blouses", "shirts")
            .replace("blouse", "shirt")
            .replace("pants", "trousers")
            .replace("skirt", "coat")
        )
        for question, expected in ((sale, 34), (commissioned, 100), (recurring, 235)):
            with self.subTest(question=question):
                result = compile_signed_events(question)
                self.assertTrue(result.ok, result.reason)
                assert result.compiled is not None
                self.assertEqual(result.compiled.solution.target_value, expected)

    def test_pronoun_scope_relation_and_item_mutations_fail_closed(self) -> None:
        mutations = (
            EXHAUSTIVE_SALE_CASE.replace("She has 6 yams", "Mira has 6 yams"),
            EXHAUSTIVE_SALE_CASE.replace("sells everything", "sells some items"),
            EXHAUSTIVE_SALE_CASE.replace(
                "4 carrots which cost", "4 carrots which weigh"
            ),
            COMMISSIONED_LEDGER_CASE.replace("his friend Ali", "her friend Ali"),
            COMMISSIONED_LEDGER_CASE.replace("pay John", "pay Mira"),
            RECURRING_RATE_CASE.replace("charge her", "charge Mira"),
            RECURRING_RATE_CASE.replace("per pair of pants", "per pair of socks"),
            RECURRING_RATE_CASE.replace("in 5 weeks", "in 5 days"),
            RECURRING_RATE_CASE + " Reference 99.",
        )
        for question in mutations:
            with self.subTest(question=question):
                self.assertFalse(compile_signed_events(question).ok)

    def test_absolute_scope_direction_and_actor_mutations_fail_closed(self) -> None:
        mutations = (
            ABSOLUTE_SCORE_CASE.replace("They receive", "Ava receives"),
            ABSOLUTE_SCORE_CASE.replace("4 seconds slower", "4 seconds later"),
            ABSOLUTE_SCORE_CASE.replace(
                "than Emma and collects", "than Talia and collects"
            ),
            ABSOLUTE_SCORE_CASE.replace(
                "difference between their two scores", "higher score Ava earned"
            ),
            ABSOLUTE_SCORE_CASE + " Reference 99.",
        )
        for question in mutations:
            with self.subTest(question=question):
                self.assertFalse(compile_signed_events(question).ok)


if __name__ == "__main__":
    unittest.main()
