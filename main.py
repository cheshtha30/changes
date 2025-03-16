def apply_logical_rules(knowledge_base):
    # 1. Alice likes maths and she likes stories
    knowledge_base["like(A, maths)"] = True
    knowledge_base["like(A, stories)"] = True

    # 2. If someone likes maths, she likes algebra
    for person in knowledge_base.keys():
        if knowledge_base.get("like({}, maths)".format(person), False):
            knowledge_base["like({}, algebra)".format(person)] = True

    # 3. If someone likes physics and algebra, she will go to college
    for person in knowledge_base.keys():
        if knowledge_base.get("like({}, physics)".format(person), False) \
                and knowledge_base.get("like({}, algebra)".format(person), False):
            knowledge_base["goes_to({}, college)".format(person)] = True

    # 4. Alice does like stories or like physics
    knowledge_base["like(A, stories)"] or knowledge_base.get("like(A, physics)", False)

    # 5. Alice does like chem and history
    not knowledge_base["like(A, chemistry)"] and not knowledge_base.get("like(A, history)", False)

    # 6. like(A, maths) [A-E] and Elimination
    like_maths = knowledge_base["like(A, maths)"]

    # 7. like(A, stories) [A -E -4A]
    like_stories = knowledge_base["like(A, stories)"]

    # 8. like(A, maths) => like(A, Algebra)[ UI -2]
    if like_maths:
        knowledge_base["like(A, algebra)"] = True

    # 9. like (A, Algebra) [modus potum alpha, alpha=> beta
    if like_maths:
        if knowledge_base["like(A, algebra)"]:
            knowledge_base["like(A, algebra)"] = True

    # 10. 4A like[(A, stories) Ʌ ~ like(A, Physics )
    if not like_stories or knowledge_base.get("like(A, physics)", False):
        knowledge_base["like(A, algebra)"] = True
        knowledge_base["like(A, physics)"] = True

    # 11. like (A, Algebra) Ʌ like( A, physics) ⊃ goes to [A, college]
    if knowledge_base.get("like(A, algebra)", False) and knowledge_base.get("like(A, physics)", False):
        knowledge_base["goes_to(A, college)"] = True

    # 12. goes to (A, college)[ 9, 10 modus point ]
    if knowledge_base.get("like(A, algebra)", False) and knowledge_base.get("like(A, physics)", False):
        knowledge_base["goes_to(A, college)"] = True

def check_if_goes_to_college(knowledge_base):
    # Check if Alice goes to college
    if knowledge_base.get("goes_to(A, college)", True):
        print("Alice goes to college.")
    else:
        print("Alice does not go to college.")

# Knowledge Base
knowledge_base = {
    "like(A, maths)": True,
    "like(A, stories)": True,
    "likes(x, maths)": True,
    "like(x, algebra)": True,
    "like(x, physics)": True,
    "goes_to(x, college)": False,
    "like(A, chemistry)": False,
    "like(A, history)": False
}

# Apply logical rules to the knowledge base
apply_logical_rules(knowledge_base)

# Check if Alice goes to college
check_if_goes_to_college(knowledge_base)
