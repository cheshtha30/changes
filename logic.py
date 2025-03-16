from logic import *

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

# Check if Alice likes maths and stories
if alice_likes_maths_and_stories(knowledge_base):
    print("Alice likes maths and stories.")

# Apply logical deduction rules
likes_algebra(knowledge_base)
goes_to_college(knowledge_base)

# Check if Alice goes to college
print(alice_goes_to_college(knowledge_base))
